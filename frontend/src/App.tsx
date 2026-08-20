import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Axis,
  FileInfo,
  Point,
  RuntimeInfo,
  getRuntime,
  listFiles,
  maskDataUrl,
  prepareSlice,
  propagate,
  segment,
  sliceUrl,
} from "./api";
import { Viewer } from "./Viewer";
import "./App.css";

interface PropagationState {
  axis: Axis;
  running: boolean;
  done: number;
  total: number;
  error?: string;
}

export default function App() {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [file, setFile] = useState<FileInfo | null>(null);
  const [axis, setAxis] = useState<Axis>("inline");
  const [index, setIndex] = useState(0);
  const [label, setLabel] = useState<0 | 1>(1);
  const [points, setPoints] = useState<Point[]>([]);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
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

  // Per-frame propagated masks for instant scrubbing.
  const propMasksRef = useRef<Map<number, string>>(new Map());
  const requestSeq = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

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
    getRuntime()
      .then(setRuntime)
      .catch(() => setRuntime(null));
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

  const resetPicks = useCallback(() => {
    abortRef.current?.abort();
    requestSeq.current += 1;
    setPoints([]);
    setMaskUrl(null);
    setCoverage(null);
    setSegmenting(false);
  }, []);

  // Reset picks and pre-encode the new slice so the first click is warm.
  useEffect(() => {
    if (!file) return;
    resetPicks();
    setPrepared(false);
    const propagated = propMasksRef.current.get(index);
    if (propagation && propagation.axis === axis && propagated) {
      setMaskUrl(maskDataUrl(propagated));
    }
    const timer = setTimeout(() => {
      prepareSlice(file.name, axis, index)
        .then(() => setPrepared(true))
        .catch(() => setPrepared(false));
    }, 250);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, axis, index]);

  const runSegment = useCallback(
    (pts: Point[]) => {
      if (!file || pts.length === 0) {
        setMaskUrl(null);
        setCoverage(null);
        return;
      }
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      const seq = ++requestSeq.current;
      const started = performance.now();
      setSegmenting(true);
      segment(file.name, axis, index, pts, controller.signal)
        .then((result) => {
          if (seq !== requestSeq.current) return; // stale response
          setMaskUrl(maskDataUrl(result.mask));
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
    [file, axis, index],
  );

  const handlePick = useCallback(
    (col: number, row: number) => {
      if (propagation?.running) return;
      const next = [...points, { col, row, label }];
      setPoints(next); // marker appears immediately
      runSegment(next);
    },
    [points, label, runSegment, propagation],
  );

  const handleUndo = useCallback(() => {
    const next = points.slice(0, -1);
    setPoints(next);
    runSegment(next);
  }, [points, runSegment]);

  const handlePropagate = useCallback(() => {
    if (!file || points.length === 0) return;
    propMasksRef.current = new Map();
    setPropagation({ axis, running: true, done: 0, total: axisCount });
    propagate(file.name, axis, index, points, (event) => {
      if (event.type === "frame") {
        propMasksRef.current.set(event.frame, event.mask);
        setPropagation({
          axis,
          running: true,
          done: event.done,
          total: event.total,
        });
        if (event.frame === index) setMaskUrl(maskDataUrl(event.mask));
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
  }, [file, axis, index, points, axisCount]);

  if (!file) {
    return <div className="app-empty">{status || "Loading..."}</div>;
  }

  const imageUrl = sliceUrl(file.name, axis, index);
  const progressPct = propagation
    ? Math.round((100 * propagation.done) / propagation.total)
    : 0;

  return (
    <div className="app">
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
          <span>Click adds</span>
          <div className="toggle-row">
            <button
              className={label === 1 ? "toggle active-pos" : "toggle"}
              onClick={() => setLabel(1)}
            >
              + object
            </button>
            <button
              className={label === 0 ? "toggle active-neg" : "toggle"}
              onClick={() => setLabel(0)}
            >
              − background
            </button>
          </div>
        </div>

        <div className="field">
          <div className="toggle-row">
            <button className="toggle" onClick={handleUndo} disabled={!points.length}>
              Undo point
            </button>
            <button className="toggle" onClick={resetPicks} disabled={!points.length}>
              Clear
            </button>
          </div>
        </div>

        {file.kind === "3d" && (
          <button
            className="primary"
            onClick={handlePropagate}
            disabled={!points.length || propagation?.running}
          >
            {propagation?.running
              ? `Propagating ${propagation.done}/${propagation.total}...`
              : `Propagate through volume (${axis})`}
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
            <span className="stat-label">Slice encoder</span>
            <span className={prepared ? "stat-ok" : "stat-wait"}>
              {prepared ? "ready" : "encoding..."}
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
      </aside>

      <main className="stage">
        <Viewer
          imageUrl={imageUrl}
          maskUrl={maskUrl}
          points={points}
          sliceWidth={sliceSize.w}
          sliceHeight={sliceSize.h}
          displayWidth={displayWidth}
          displayHeight={displayHeight}
          maskOpacity={maskOpacity}
          busy={segmenting}
          onPick={handlePick}
        />
      </main>
    </div>
  );
}
