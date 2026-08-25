import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Axis,
  FileInfo,
  Point,
  RuntimeInfo,
  getRuntime,
  listFiles,
  maskDataUrl,
  objectColor,
  prepareSlice,
  propagate,
  segment,
  sliceUrl,
} from "./api";
import { Viewer, ViewerObject } from "./Viewer";
import "./App.css";

interface PropagationState {
  axis: Axis;
  running: boolean;
  done: number;
  total: number;
  error?: string;
}

interface SegObject extends ViewerObject {
  id: number;
  points: Point[];
  maskUrl: string | null;
}

const freshObject = (id: number): SegObject => ({ id, points: [], maskUrl: null });

export default function App() {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [file, setFile] = useState<FileInfo | null>(null);
  const [axis, setAxis] = useState<Axis>("inline");
  const [index, setIndex] = useState(0);
  const [objects, setObjects] = useState<SegObject[]>([freshObject(0)]);
  const [activeObjectId, setActiveObjectId] = useState(0);
  const nextObjectId = useRef(1);
  const [propMaskUrl, setPropMaskUrl] = useState<string | null>(null);
  const [coverage, setCoverage] = useState<number | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [segmenting, setSegmenting] = useState(false);
  const [prepared, setPrepared] = useState(false);
  const [displayWidth, setDisplayWidth] = useState(760);
  const [displayHeight, setDisplayHeight] = useState(700);
  const [maskOpacity, setMaskOpacity] = useState(0.9);
  const [propagation, setPropagation] = useState<PropagationState | null>(null);
  const [status, setStatus] = useState("Connecting to inference server...");
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const pointReady = Boolean(runtime?.point_loaded);
  const videoReady = Boolean(runtime?.video_loaded);

  // Per-frame propagated masks for instant scrubbing.
  const propMasksRef = useRef<Map<number, string>>(new Map());
  // Per-object request bookkeeping so refreshing one object's preview
  // doesn't cancel another object's in-flight request.
  const requestSeq = useRef<Map<number, number>>(new Map());
  const abortMap = useRef<Map<number, AbortController>>(new Map());
  // Latest objects for async callbacks (slice-change re-segmentation).
  const objectsRef = useRef<SegObject[]>([]);

  useEffect(() => {
    listFiles()
      .then((f) => {
        setFiles(f);
        const preferred = f.find((x) => x.kind === "3d") ?? f[0] ?? null;
        setFile(preferred);
        setStatus(f.length ? "" : "No .sgy files found in data/");
      })
      .catch(() =>
        setStatus("Cannot reach the inference server. Start it with: uvicorn server.main:app"),
      );
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const poll = (delayMs: number) => {
      timer = window.setTimeout(async () => {
        try {
          const info = await getRuntime();
          if (cancelled) return;
          setRuntime(info);
          const finished =
            info.load_stage === "ready" || info.load_stage === "error";
          poll(finished ? 8000 : 600);
        } catch {
          if (cancelled) return;
          setRuntime(null);
          poll(1000);
        }
      }, delayMs);
    };
    poll(0);
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, []);

  const axisCount = file ? file.axes[axis] : 1;
  const sliceSize = useMemo(() => {
    if (!file) return { w: 1, h: 1 };
    const [nIl, nXl, nS] = file.shape;
    // 2D lines are stored (n_samples, n_traces): time down, traces across.
    if (file.kind === "2d") return { w: file.shape[1], h: file.shape[0] };
    if (axis === "inline") return { w: nXl, h: nS };
    if (axis === "crossline") return { w: nIl, h: nS };
    return { w: nXl, h: nIl };
  }, [file, axis]);

  objectsRef.current = objects;

  const abortAll = useCallback(() => {
    abortMap.current.forEach((c) => c.abort());
    abortMap.current.clear();
    requestSeq.current = new Map();
  }, []);

  const resetPicks = useCallback(() => {
    abortAll();
    setObjects([freshObject(0)]);
    setActiveObjectId(0);
    nextObjectId.current = 1;
    setPropMaskUrl(null);
    setCoverage(null);
    setSegmenting(false);
  }, [abortAll]);

  // A new file or axis invalidates all picks; a slice change does NOT -
  // objects persist so you can refine them on any slice (SAM2-style).
  useEffect(() => {
    resetPicks();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, axis]);

  const setObjectMask = useCallback((objectId: number, url: string | null) => {
    setObjects((prev) =>
      prev.map((o) => (o.id === objectId ? { ...o, maskUrl: url } : o)),
    );
  }, []);

  // Preview one object's mask from its points on the CURRENT slice only.
  // Other objects' requests are untouched (embedding is shared server-side).
  const runSegment = useCallback(
    (objectId: number, pts: Point[]) => {
      // Negative-only clicks can't produce a standalone preview (there is
      // nothing positive to segment from). They are stored as refinement
      // prompts and take effect on re-propagation, where the tracker's
      // memory of the object gives them something to subtract from.
      if (!file || pts.length === 0 || !pts.some((p) => p.label === 1)) {
        setObjectMask(objectId, null);
        return;
      }
      abortMap.current.get(objectId)?.abort();
      const controller = new AbortController();
      abortMap.current.set(objectId, controller);
      const seq = (requestSeq.current.get(objectId) ?? 0) + 1;
      requestSeq.current.set(objectId, seq);
      const started = performance.now();
      setSegmenting(true);
      segment(file.name, axis, index, pts, objectId, controller.signal)
        .then((result) => {
          if (seq !== requestSeq.current.get(objectId)) return; // stale
          setObjectMask(objectId, maskDataUrl(result.mask));
          setCoverage(result.coverage);
          setLatencyMs(performance.now() - started);
          setSegmenting(false);
        })
        .catch((err) => {
          if (controller.signal.aborted) return;
          setSegmenting(false);
          setStatus(`Segmentation failed: ${err.message}`);
        });
    },
    [file, axis, index, setObjectMask],
  );

  // On slice change: keep every object and its points, drop the stale
  // single-slice previews, pre-encode the slice, then re-preview objects
  // that have points here.
  useEffect(() => {
    if (!file || !pointReady) return;
    abortAll();
    setSegmenting(false);
    setObjects((prev) => prev.map((o) => ({ ...o, maskUrl: null })));
    setPropMaskUrl(null);
    const propagated = propMasksRef.current.get(index);
    if (propagation && propagation.axis === axis && propagated) {
      setPropMaskUrl(maskDataUrl(propagated));
    }
    setPrepared(false);
    const timer = setTimeout(() => {
      prepareSlice(file.name, axis, index)
        .then(() => {
          setPrepared(true);
          for (const obj of objectsRef.current) {
            const pts = obj.points.filter((p) => p.slice === index);
            if (pts.length) runSegment(obj.id, pts);
          }
        })
        .catch(() => setPrepared(false));
    }, 250);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, axis, index, pointReady]);

  const handlePick = useCallback(
    (col: number, row: number, label: 0 | 1) => {
      if (propagation?.running || !prepared || !pointReady) return;
      const active = objects.find((o) => o.id === activeObjectId);
      if (!active) return;
      const nextPts = [...active.points, { col, row, label, slice: index }];
      setObjects((prev) =>
        prev.map((o) => (o.id === activeObjectId ? { ...o, points: nextPts } : o)),
      );
      runSegment(
        activeObjectId,
        nextPts.filter((p) => p.slice === index),
      );
    },
    [objects, activeObjectId, index, runSegment, propagation, prepared, pointReady],
  );

  // Undo removes the active object's most recent point on THIS slice.
  const handleUndo = useCallback(() => {
    const active = objects.find((o) => o.id === activeObjectId);
    if (!active) return;
    let removeAt = -1;
    for (let i = active.points.length - 1; i >= 0; i--) {
      if (active.points[i].slice === index) {
        removeAt = i;
        break;
      }
    }
    if (removeAt < 0) return;
    const nextPts = active.points.filter((_, i) => i !== removeAt);
    setObjects((prev) =>
      prev.map((o) => (o.id === activeObjectId ? { ...o, points: nextPts } : o)),
    );
    runSegment(
      activeObjectId,
      nextPts.filter((p) => p.slice === index),
    );
  }, [objects, activeObjectId, index, runSegment]);

  const handleAddObject = useCallback(() => {
    const id = nextObjectId.current++;
    setObjects((prev) => [...prev, freshObject(id)]);
    setActiveObjectId(id);
  }, []);

  const handleRemoveObject = useCallback(
    (objectId: number) => {
      abortMap.current.get(objectId)?.abort();
      abortMap.current.delete(objectId);
      setObjects((prev) => {
        const next = prev.filter((o) => o.id !== objectId);
        if (next.length === 0) {
          const id = nextObjectId.current++;
          setActiveObjectId(id);
          return [freshObject(id)];
        }
        if (objectId === activeObjectId) setActiveObjectId(next[0].id);
        return next;
      });
    },
    [activeObjectId],
  );

  // Only objects with at least one + point can be tracked; negative-only
  // objects have nothing to segment.
  const objectsWithPoints = useMemo(
    () => objects.filter((o) => o.points.some((p) => p.label === 1)),
    [objects],
  );
  const totalPoints = useMemo(
    () => objects.reduce((n, o) => n + o.points.length, 0),
    [objects],
  );
  const activePointsHere = useMemo(() => {
    const active = objects.find((o) => o.id === activeObjectId);
    return active ? active.points.filter((p) => p.slice === index).length : 0;
  }, [objects, activeObjectId, index]);
  // True when the active object's clicks on this slice are all negative:
  // no live preview is possible, the clicks apply on re-propagation.
  const negativeOnlyHere = useMemo(() => {
    const active = objects.find((o) => o.id === activeObjectId);
    if (!active) return false;
    const here = active.points.filter((p) => p.slice === index);
    return here.length > 0 && !here.some((p) => p.label === 1);
  }, [objects, activeObjectId, index]);
  // The viewer only shows markers and previews belonging to this slice.
  const viewerObjects = useMemo(
    () =>
      objects.map((o) => ({
        ...o,
        points: o.points.filter((p) => p.slice === index),
      })),
    [objects, index],
  );

  const handlePropagate = useCallback(() => {
    if (!file || objectsWithPoints.length === 0) return;
    propMasksRef.current = new Map();
    // The propagated overlay carries all objects; drop the per-object
    // single-slice previews so they don't double-tint the anchor frame.
    setObjects((prev) => prev.map((o) => ({ ...o, maskUrl: null })));
    setPropagation({ axis, running: true, done: 0, total: axisCount });
    propagate(file.name, axis, index, objectsWithPoints, (event) => {
      if (event.type === "frame") {
        propMasksRef.current.set(event.frame, event.mask);
        setPropagation({
          axis,
          running: true,
          done: event.done,
          total: event.total,
        });
        if (event.frame === index) setPropMaskUrl(maskDataUrl(event.mask));
      } else if (event.type === "done") {
        setPropagation((prev) =>
          prev ? { ...prev, running: false } : null,
        );
      } else {
        setPropagation((prev) =>
          prev ? { ...prev, running: false, error: event.message } : null,
        );
      }
    }).catch((err) =>
      setPropagation((prev) =>
        prev ? { ...prev, running: false, error: err.message } : null,
      ),
    );
  }, [file, axis, index, objectsWithPoints, axisCount]);

  if (!file) {
    return <div className="app-empty">{status || "Loading..."}</div>;
  }

  const imageUrl = sliceUrl(file.name, axis, index);
  const progressPct = propagation
    ? Math.round((100 * propagation.done) / propagation.total)
    : 0;
  const modelsLoading = !pointReady;
  const loadMessage = runtime?.load_error && !pointReady
    ? runtime.load_error
    : (runtime?.load_stage as string | undefined) ?? "Connecting to inference server...";

  return (
    <div className="app">
      {(modelsLoading || (runtime?.load_error && !pointReady)) && (
        <div className="loading-overlay" role="status">
          <div className="loading-card">
            {!(runtime?.load_error && !pointReady) && <div className="loading-spinner" />}
            <h2>{runtime?.load_error && !pointReady ? "Model failed to load" : "Loading SAM 3"}</h2>
            <p>{loadMessage}</p>
            {!(runtime?.load_error && !pointReady) && (
              <p className="loading-hint">
                The tracker is loaded once at startup so the first click stays fast.
                This can take a minute on the first run.
              </p>
            )}
          </div>
        </div>
      )}
      <aside className="sidebar">
        <h1>Seismic SAM</h1>
        <p className="subtitle">Interactive point segmentation</p>

        <label className="field">
          <span>Seismic file</span>
          <select
            value={file.name}
            onChange={(e) => {
              const next = files.find((f) => f.name === e.target.value);
              if (next) {
                setFile(next);
                setIndex(0);
                propMasksRef.current = new Map();
                setPropagation(null);
              }
            }}
          >
            {files.map((f) => (
              <option key={f.name}>{f.name}</option>
            ))}
          </select>
        </label>

        {file.kind === "3d" && (
          <>
            <label className="field">
              <span>Slice axis</span>
              <select
                value={axis}
                onChange={(e) => {
                  setAxis(e.target.value as Axis);
                  setIndex(0);
                }}
              >
                <option value="inline">Inline</option>
                <option value="crossline">Crossline</option>
                <option value="time">Time slice</option>
              </select>
            </label>
            <label className="field">
              <span>
                Slice {index + 1} / {axisCount}
                {propagation && propagation.axis === axis && (
                  <em className="hint"> — scrub to review propagation</em>
                )}
              </span>
              <input
                type="range"
                min={0}
                max={axisCount - 1}
                value={index}
                onChange={(e) => setIndex(Number(e.target.value))}
              />
            </label>
          </>
        )}

        <div className="field">
          <span>Clicks</span>
          <div className="toggle-row">
            <div className="click-hint active-pos" title="Left click">
              Left + object
            </div>
            <div className="click-hint active-neg" title="Right click">
              Right − background
            </div>
          </div>
        </div>

        <div className="field">
          <span>Objects</span>
          <div className="object-list">
            {objects.map((obj) => (
              <div
                key={obj.id}
                className={
                  obj.id === activeObjectId ? "object-row active" : "object-row"
                }
                onClick={() => setActiveObjectId(obj.id)}
              >
                <span
                  className="object-swatch"
                  style={{ background: objectColor(obj.id) }}
                />
                <span className="object-name">Object {obj.id + 1}</span>
                <span
                  className="object-count"
                  title="points on this slice / total points"
                >
                  {obj.points.filter((p) => p.slice === index).length}/
                  {obj.points.length} pts
                </span>
                <button
                  className="object-remove"
                  title="Remove object"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemoveObject(obj.id);
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
          <button className="toggle" onClick={handleAddObject}>
            + Add object
          </button>
          {file.kind === "3d" && (
            <p className="hint">
              Points persist across slices — scrub to another slice to add
              refinement clicks, then re-propagate.
            </p>
          )}
          {negativeOnlyHere && (
            <p className="hint hint-notice">
              Only − points on this slice: they refine the tracked mask on
              re-propagation. No live preview without a + point here.
            </p>
          )}
        </div>

        <div className="field">
          <div className="toggle-row">
            <button
              className="toggle"
              onClick={handleUndo}
              disabled={!activePointsHere}
              title="Removes the active object's last point on this slice"
            >
              Undo point
            </button>
            <button className="toggle" onClick={resetPicks} disabled={!totalPoints}>
              Clear all
            </button>
          </div>
        </div>

        {file.kind === "3d" && (
          <button
            className="primary"
            onClick={handlePropagate}
            disabled={!objectsWithPoints.length || propagation?.running || !videoReady}
          >
            {propagation?.running
              ? `Propagating ${propagation.done}/${propagation.total}...`
              : !videoReady
                ? "Loading volume tracker..."
                : `${propagation ? "Re-propagate" : "Propagate"} ${
                    objectsWithPoints.length || ""
                  } object${objectsWithPoints.length === 1 ? "" : "s"} (${axis})`}
          </button>
        )}
        {propagation && (
          <div className="progress-wrap">
            <div className="progress-bar" style={{ width: `${progressPct}%` }} />
          </div>
        )}
        {propagation?.error && <p className="error">{propagation.error}</p>}

        <details className="display-settings">
          <summary>Display</summary>
          <label className="field">
            <span>Width {displayWidth}px</span>
            <input
              type="range"
              min={300}
              max={1400}
              step={20}
              value={displayWidth}
              onChange={(e) => setDisplayWidth(Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>Height {displayHeight}px</span>
            <input
              type="range"
              min={300}
              max={2000}
              step={20}
              value={displayHeight}
              onChange={(e) => setDisplayHeight(Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>Mask opacity</span>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={maskOpacity}
              onChange={(e) => setMaskOpacity(Number(e.target.value))}
            />
          </label>
        </details>

        <details
          className="display-settings"
          onToggle={(event) => {
            if ((event.currentTarget as HTMLDetailsElement).open) {
              getRuntime().then(setRuntime).catch(() => setRuntime(null));
            }
          }}
        >
          <summary>Runtime</summary>
          {runtime ? (
            <dl className="runtime-list">
              <dt>Hardware</dt>
              <dd>{runtime.hardware.device_name}</dd>
              <dt>CUDA</dt>
              <dd>
                {runtime.hardware.cuda_available
                  ? `yes (${runtime.hardware.cuda_version ?? "unknown"}, sm ${runtime.hardware.compute_capability ?? "?"})`
                  : "no — CPU"}
              </dd>
              {runtime.hardware.vram.total_gb != null && (
                <>
                  <dt>VRAM</dt>
                  <dd>
                    {runtime.hardware.vram.allocated_gb ?? "?"} /{" "}
                    {runtime.hardware.vram.total_gb} GB used
                  </dd>
                </>
              )}
              <dt>Architecture</dt>
              <dd>{runtime.architecture}</dd>
              <dt>Checkpoint</dt>
              <dd className="runtime-mono">{runtime.checkpoint}</dd>
              <dt>Point model</dt>
              <dd>
                {runtime.point_model}
                {runtime.point_loaded
                  ? ` — loaded on ${runtime.point_device ?? "?"}`
                  : " — not loaded yet"}
              </dd>
              <dt>Video model</dt>
              <dd>
                {runtime.video_model}
                {runtime.video_loaded
                  ? ` — loaded on ${runtime.video_device ?? "?"}${runtime.video_precision ? `, ${runtime.video_precision}` : ""}`
                  : " — not loaded yet"}
              </dd>
              <dt>Embedding cache</dt>
              <dd>
                {runtime.cached_slices}/{runtime.embedding_cache_size ?? "?"} slices
              </dd>
              <dt>PyTorch</dt>
              <dd>{runtime.software.torch}</dd>
              <dt>Transformers</dt>
              <dd>{runtime.software.transformers ?? "unknown"}</dd>
              <dt>Python</dt>
              <dd>{runtime.software.python}</dd>
              {file && (
                <>
                  <dt>Active file</dt>
                  <dd>
                    {file.name} ({file.kind}, {file.shape.join(" × ")})
                  </dd>
                </>
              )}
            </dl>
          ) : (
            <p className="hint">Runtime details unavailable until the API is reachable.</p>
          )}
        </details>

        <div className="stats">
          <div>
            <span className="stat-label">Point tracker</span>
            <span className={pointReady ? "stat-ok" : "stat-wait"}>
              {pointReady ? "ready" : (runtime?.load_stage ?? "loading...")}
            </span>
          </div>
          <div>
            <span className="stat-label">Volume tracker</span>
            <span className={videoReady ? "stat-ok" : "stat-wait"}>
              {videoReady ? "ready" : pointReady ? "loading..." : "waiting"}
            </span>
          </div>
          <div>
            <span className="stat-label">Slice encoder</span>
            <span className={prepared ? "stat-ok" : "stat-wait"}>
              {prepared ? "ready" : pointReady ? "encoding..." : "waiting"}
            </span>
          </div>
          {latencyMs !== null && (
            <div>
              <span className="stat-label">Last click → mask</span>
              <span>{latencyMs < 1000 ? `${Math.round(latencyMs)} ms` : `${(latencyMs / 1000).toFixed(2)} s`}</span>
            </div>
          )}
          {coverage !== null && (
            <div>
              <span className="stat-label">Mask coverage</span>
              <span>{(100 * coverage).toFixed(2)}%</span>
            </div>
          )}
        </div>
        {status && <p className="error">{status}</p>}
        {runtime?.load_error && pointReady && (
          <p className="error">{runtime.load_error}</p>
        )}
      </aside>

      <main className="stage">
        <Viewer
          imageUrl={imageUrl}
          propagatedMaskUrl={propMaskUrl}
          objects={viewerObjects}
          activeObjectId={activeObjectId}
          sliceWidth={sliceSize.w}
          sliceHeight={sliceSize.h}
          displayWidth={displayWidth}
          displayHeight={displayHeight}
          maskOpacity={maskOpacity}
          busy={segmenting || !prepared || !pointReady}
          onPick={handlePick}
        />
      </main>
    </div>
  );
}
