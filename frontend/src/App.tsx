import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Axis,
  ExportResult,
  FileInfo,
  ModelFamily,
  Point,
  PropagationEvent,
  RuntimeInfo,
  TextCheckpointInfo,
  downloadExportedVolume,
  exportVolume,
  getRuntime,
  listFiles,
  maskDataUrl,
  objectColor,
  prepareSlice,
  propagate,
  refine,
  resweep,
  segment,
  autoSegment,
  setModel,
  setTextCheckpoint,
  sliceUrl,
} from "./api";
import { Viewer, ViewerObject } from "./Viewer";
import "./App.css";

interface PropagationState {
  axis: Axis;
  running: boolean;
  done: number;
  total: number;
  /** True while re-tracking from edits rather than rebuilding. */
  resweeping?: boolean;
  error?: string;
}

interface SegObject extends ViewerObject {
  id: number;
  points: Point[];
  maskUrl: string | null;
}

const freshObject = (id: number): SegObject => ({ id, points: [], maskUrl: null });

const FALLBACK_TRACKERS = [
  { id: "sam3" as const, label: "SAM 3", checkpoint: "facebook/sam3", gated: true },
  { id: "sam2" as const, label: "SAM 2", checkpoint: "facebook/sam2.1-hiera-large", gated: false },
];

const FALLBACK_TEXT_CHECKPOINTS: TextCheckpointInfo[] = [
  {
    id: "facebook/sam3",
    label: "SAM 3 (official)",
    path: "facebook/sam3",
    source: "official",
  },
];

function matchTextCheckpoint(
  current: string | null,
  options: TextCheckpointInfo[],
): string {
  if (!options.length) return current ?? "";
  if (!current) return options[0].path;
  const exact = options.find((item) => item.path === current);
  if (exact) return exact.path;
  const normalized = current.replace(/\\/g, "/").toLowerCase();
  const fuzzy = options.find((item) => {
    const path = item.path.replace(/\\/g, "/").toLowerCase();
    return path === normalized || item.id.toLowerCase() === normalized;
  });
  return fuzzy?.path ?? current;
}

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
  // Slice edits applied to the live tracker but not yet swept outward.
  const [pendingEdits, setPendingEdits] = useState(0);
  // Set when a prompt was removed from the live tracker, which it cannot
  // undo in place - only a full re-propagation drops it.
  const [promptsDropped, setPromptsDropped] = useState(false);
  const [includeAmplitude, setIncludeAmplitude] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [exportInfo, setExportInfo] = useState<ExportResult | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  // Object ids the last completed propagation actually tracked. Kept in
  // React state (not only the runtime poll) so a stale /api/runtime
  // response during a long sweep cannot hide export or disable refine.
  const [trackedObjectIds, setTrackedObjectIds] = useState<number[]>([]);
  const [modelFamily, setModelFamily] = useState<ModelFamily>("sam3");
  const [autoMaskUrl, setAutoMaskUrl] = useState<string | null>(null);
  const [autoCoverage, setAutoCoverage] = useState<number | null>(null);
  const [autoDetecting, setAutoDetecting] = useState(false);
  const [autoError, setAutoError] = useState<string | null>(null);
  const [autoCheckpoint, setAutoCheckpoint] = useState<string | null>(null);
  const [textCheckpoint, setTextCheckpointPath] = useState<string | null>(null);
  const userPickedTextCheckpoint = useRef(false);
  const pointReady = Boolean(runtime?.point_loaded) && runtime?.family === modelFamily;
  const videoReady = Boolean(runtime?.video_loaded) && runtime?.family === modelFamily;

  // Per-frame propagated masks for instant scrubbing.
  const propMasksRef = useRef<Map<number, string>>(new Map());
  // Per-object request bookkeeping so refreshing one object's preview
  // doesn't cancel another object's in-flight request.
  const requestSeq = useRef<Map<number, number>>(new Map());
  const abortMap = useRef<Map<number, AbortController>>(new Map());
  // Latest objects for async callbacks (slice-change re-segmentation).
  const objectsRef = useRef<SegObject[]>([]);
  // Objects the live tracker holds, for use inside effects.
  const trackedIdsRef = useRef<Set<number>>(new Set());
  const trackedReadyRef = useRef(false);
  const userPickedFamily = useRef<ModelFamily | null>(null);

  useEffect(() => {
    let cancelled = false;

    // NPY metadata is effectively instant. Show those files immediately
    // instead of waiting for SEG-Y geometry scans over every trace header.
    listFiles("npy")
      .then((npyFiles) => {
        if (cancelled || !npyFiles.length) return;
        setFiles(npyFiles);
        setFile((current) => current ?? npyFiles.find((x) => x.kind === "3d") ?? npyFiles[0]);
        setStatus("");
      })
      .catch(() => undefined);

    listFiles()
      .then((f) => {
        if (cancelled) return;
        setFiles(f);
        setFile((current) => {
          if (current) return f.find((item) => item.name === current.name) ?? current;
          return f.find((x) => x.kind === "3d") ?? f[0] ?? null;
        });
        setStatus(f.length ? "" : "No .sgy or .npy files found in data/");
      })
      .catch(() => {
        if (!cancelled) {
          setStatus("Cannot reach the inference server. Start it with: uvicorn server.main:app");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const poll = (delayMs: number) => {
      timer = window.setTimeout(async () => {
        try {
          const info = await getRuntime();
          if (cancelled) return;
          setRuntime((prev) => {
            // A poll that started while a sweep had cleared the server's
            // volume must not wipe the one that just finished.
            if (
              !info.tracked_volume &&
              prev?.tracked_volume &&
              trackedReadyRef.current
            ) {
              return { ...info, tracked_volume: prev.tracked_volume };
            }
            return info;
          });
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
  }, [modelFamily]);

  useEffect(() => {
    if (!runtime?.family || userPickedFamily.current != null) return;
    setModelFamily(runtime.family);
  }, [runtime?.family]);

  useEffect(() => {
    if (!runtime?.text_checkpoint || userPickedTextCheckpoint.current) return;
    setTextCheckpointPath(runtime.text_checkpoint);
  }, [runtime?.text_checkpoint]);

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

  // Only objects this UI session actually propagated are "live". Do not
  // fall back to runtime.tracked_volume: that leftover from an earlier
  // run would send the first clicks to /api/refine and skip the
  // single-slice preview, leaving points with no mask.
  const liveTracked = useMemo(() => {
    const tracked = runtime?.tracked_volume;
    if (!tracked || !file) return null;
    return tracked.file === file.name && tracked.axis === axis ? tracked : null;
  }, [runtime, file, axis]);
  const trackedIds = trackedObjectIds;
  trackedIdsRef.current = new Set(trackedIds);
  trackedReadyRef.current = trackedIds.length > 0;

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
    // Drop the propagated overlay too, otherwise scrubbing keeps showing
    // masks for objects that no longer exist.
    propMasksRef.current = new Map();
    setPropagation(null);
    setPendingEdits(0);
    setPromptsDropped(true);
    setTrackedObjectIds([]);
    setExportInfo(null);
    setExportError(null);
    setAutoMaskUrl(null);
    setAutoCoverage(null);
    setAutoError(null);
  }, [abortAll]);

  const handleModelChange = useCallback(
    (family: ModelFamily) => {
      if (family === modelFamily || propagation?.running) return;
      userPickedFamily.current = family;
      setModelFamily(family);
      abortAll();
      setPropMaskUrl(null);
      setCoverage(null);
      setSegmenting(false);
      setPrepared(false);
      propMasksRef.current = new Map();
      setPropagation(null);
      setPendingEdits(0);
      setPromptsDropped(true);
      setTrackedObjectIds([]);
      setExportInfo(null);
      setExportError(null);
      setObjects((prev) => prev.map((o) => ({ ...o, maskUrl: null })));
      setModel(family)
        .then(() => getRuntime().then(setRuntime).catch(() => undefined))
        .catch((err: Error) => {
          setStatus(`Could not switch tracker: ${err.message}`);
          const fallback = runtime?.family ?? "sam3";
          userPickedFamily.current = fallback;
          setModelFamily(fallback);
        });
    },
    [modelFamily, propagation?.running, abortAll, runtime?.family],
  );

  const handleTextCheckpointChange = useCallback(
    (checkpoint: string) => {
      if (!checkpoint || checkpoint === textCheckpoint) return;
      userPickedTextCheckpoint.current = true;
      const previous = textCheckpoint;
      setTextCheckpointPath(checkpoint);
      setAutoMaskUrl(null);
      setAutoCoverage(null);
      setAutoError(null);
      setAutoCheckpoint(null);
      setTextCheckpoint(checkpoint)
        .then((result) => {
          setTextCheckpointPath(result.checkpoint);
          return getRuntime().then(setRuntime);
        })
        .catch((err: Error) => {
          setStatus(`Could not switch checkpoint: ${err.message}`);
          userPickedTextCheckpoint.current = false;
          setTextCheckpointPath(previous);
        });
    },
    [textCheckpoint],
  );

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

  /** Start a per-object request, cancelling that object's previous one. */
  const beginRequest = useCallback((objectId: number) => {
    abortMap.current.get(objectId)?.abort();
    const controller = new AbortController();
    abortMap.current.set(objectId, controller);
    const seq = (requestSeq.current.get(objectId) ?? 0) + 1;
    requestSeq.current.set(objectId, seq);
    const isCurrent = () => seq === requestSeq.current.get(objectId);
    return { controller, isCurrent, started: performance.now() };
  }, []);

  // Preview one object's mask from its points on the CURRENT slice only.
  // Other objects' requests are untouched (embedding is shared server-side).
  const runSegment = useCallback(
    (objectId: number, pts: Point[]) => {
      // Without a tracked volume there is no memory of the object here, so
      // negative-only clicks have nothing to subtract from and cannot
      // produce a mask on their own.
      if (!file || pts.length === 0 || !pts.some((p) => p.label === 1)) {
        setObjectMask(objectId, null);
        return;
      }
      const { controller, isCurrent, started } = beginRequest(objectId);
      setSegmenting(true);
      segment(file.name, axis, index, pts, objectId, controller.signal)
        .then((result) => {
          if (!isCurrent()) return; // stale
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
    [file, axis, index, setObjectMask, beginRequest],
  );

  // Edit a tracked object on this slice: the tracker combines the clicks
  // with its memory of the object here, so a − click carves the
  // propagated mask and a + click grows it, both visible immediately.
  const runRefine = useCallback(
    (objectId: number, pts: Point[]) => {
      if (!file || pts.length === 0) return;
      const { controller, isCurrent, started } = beginRequest(objectId);
      const slice = index;
      setSegmenting(true);
      refine(file.name, axis, slice, pts, objectId, controller.signal)
        .then((result) => {
          if (!isCurrent()) return; // stale
          propMasksRef.current.set(slice, result.mask);
          setPropMaskUrl(maskDataUrl(result.mask));
          setCoverage(result.coverage);
          setLatencyMs(performance.now() - started);
          setSegmenting(false);
          setPendingEdits((n) => n + 1);
        })
        .catch(() => {
          if (controller.signal.aborted) return;
          setSegmenting(false);
          runSegment(objectId, pts);
        });
    },
    [file, axis, index, beginRequest, runSegment],
  );

  // Route a slice's clicks to whichever path can actually render them.
  const applyClicks = useCallback(
    (objectId: number, ptsHere: Point[]) => {
      if (!trackedIds.includes(objectId)) {
        runSegment(objectId, ptsHere);
        return;
      }
      if (ptsHere.length) {
        runRefine(objectId, ptsHere);
      } else {
        // The live session cannot un-prompt a slice; only a full rebuild
        // forgets these clicks.
        setPromptsDropped(true);
      }
    },
    [trackedIds, runSegment, runRefine],
  );

  useEffect(() => {
    setAutoMaskUrl(null);
    setAutoCoverage(null);
    setAutoError(null);
  }, [file?.name, axis, index]);

  const handleAutoDetect = useCallback(() => {
    if (!file || autoDetecting) return;
    setAutoDetecting(true);
    setAutoError(null);
    autoSegment(file.name, axis, index)
      .then((result) => {
        setAutoMaskUrl(maskDataUrl(result.mask));
        setAutoCoverage(result.coverage);
        setAutoCheckpoint(result.checkpoint);
        if (result.unloaded_tracker) {
          setStatus(
            "Facies detector needed the GPU, so the click tracker was unloaded. Switch SAM 3 / SAM 2 to reload it.",
          );
        }
      })
      .catch((err: Error) => setAutoError(err.message))
      .finally(() => setAutoDetecting(false));
  }, [file, axis, index, autoDetecting]);

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
            // Tracked objects already show through the propagated overlay;
            // only skip the single-slice decoder when that overlay exists.
            if (
              trackedIdsRef.current.has(obj.id) &&
              propMasksRef.current.has(index)
            ) {
              continue;
            }
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
      if (propagation?.running || !pointReady) return;
      // Refinement talks to the live tracker and does not need the
      // single-slice encoder. Only untracked objects wait for prepare.
      const tracked = trackedIdsRef.current.has(activeObjectId);
      if (!prepared && !tracked) return;
      const active = objectsRef.current.find((o) => o.id === activeObjectId);
      if (!active) return;
      const nextPts = [...active.points, { col, row, label, slice: index }];
      const next = objectsRef.current.map((o) =>
        o.id === activeObjectId ? { ...o, points: nextPts } : o,
      );
      objectsRef.current = next;
      setObjects(next);
      applyClicks(
        activeObjectId,
        nextPts.filter((p) => p.slice === index),
      );
    },
    [activeObjectId, index, applyClicks, propagation, prepared, pointReady],
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
    applyClicks(
      activeObjectId,
      nextPts.filter((p) => p.slice === index),
    );
  }, [objects, activeObjectId, index, applyClicks]);

  const handleAddObject = useCallback(() => {
    const id = nextObjectId.current++;
    setObjects((prev) => [...prev, freshObject(id)]);
    setActiveObjectId(id);
  }, []);

  const handleRemoveObject = useCallback(
    (objectId: number) => {
      abortMap.current.get(objectId)?.abort();
      abortMap.current.delete(objectId);
      // The live tracker still holds this object; only a full rebuild
      // stops tracking it.
      if (trackedIdsRef.current.has(objectId)) setPromptsDropped(true);
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
  // Objects the live tracker does not know about yet (new since the last
  // propagation), which forces a full rebuild rather than a re-sweep.
  const untrackedObjects = useMemo(
    () => objectsWithPoints.filter((o) => !trackedIds.includes(o.id)),
    [objectsWithPoints, trackedIds],
  );
  const needsFullPropagate =
    trackedIds.length === 0 || untrackedObjects.length > 0 || promptsDropped;
  // Without a live tracker, negative-only clicks on a slice cannot be
  // rendered at all; with one they refine the tracked mask directly.
  const negativeOnlyHere = useMemo(() => {
    const active = objects.find((o) => o.id === activeObjectId);
    if (!active || trackedIds.includes(activeObjectId)) return false;
    const here = active.points.filter((p) => p.slice === index);
    return here.length > 0 && !here.some((p) => p.label === 1);
  }, [objects, activeObjectId, index, trackedIds]);
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
    // A re-sweep only re-tracks from the edited slices, reusing the live
    // sessions; anything else has to rebuild them from all the prompts.
    const reuseSessions = !needsFullPropagate && pendingEdits > 0;
    if (!reuseSessions) {
      propMasksRef.current = new Map();
      setTrackedObjectIds([]);
      // The propagated overlay carries all objects; drop the per-object
      // single-slice previews so they don't double-tint the anchor frame.
      setObjects((prev) => prev.map((o) => ({ ...o, maskUrl: null })));
    }
    setPropagation({
      axis,
      running: true,
      done: 0,
      total: axisCount,
      resweeping: reuseSessions,
    });
    setExportInfo(null);

    const onEvent = (event: PropagationEvent) => {
      if (event.type === "frame") {
        propMasksRef.current.set(event.frame, event.mask);
        setPropagation({
          axis,
          running: true,
          done: event.done,
          total: event.total,
          resweeping: reuseSessions,
        });
        if (event.frame === index) setPropMaskUrl(maskDataUrl(event.mask));
      } else if (event.type === "done") {
        setPropagation((prev) => (prev ? { ...prev, running: false } : null));
        setPendingEdits(0);
        setPromptsDropped(false);
        setTrackedObjectIds(objectsWithPoints.map((o) => o.id));
        // Pick up the new tracked-volume state without waiting for the poll.
        getRuntime().then(setRuntime).catch(() => undefined);
      } else {
        setPropagation((prev) =>
          prev ? { ...prev, running: false, error: event.message } : null,
        );
      }
    };

    const request = reuseSessions
      ? resweep(file.name, axis, onEvent)
      : propagate(file.name, axis, index, objectsWithPoints, onEvent);
    request.catch((err) =>
      setPropagation((prev) =>
        prev ? { ...prev, running: false, error: err.message } : null,
      ),
    );
  }, [
    file,
    axis,
    index,
    objectsWithPoints,
    axisCount,
    needsFullPropagate,
    pendingEdits,
  ]);

  const canExport = trackedIds.length > 0 || Boolean(liveTracked);

  const handleExport = useCallback(() => {
    if (!file || !canExport) return;
    setExporting(true);
    setExportError(null);
    setExportInfo(null);
    exportVolume(file.name, axis, includeAmplitude)
      .then(async (info) => {
        setExportInfo(info);
        const filename = info.path.replace(/^.*[\\/]/, "") || `${file.name}.vti`;
        await downloadExportedVolume(file.name, axis, filename);
        setExporting(false);
      })
      .catch((err) => {
        setExportError(err.message);
        setExporting(false);
      });
  }, [file, axis, canExport, includeAmplitude]);

  if (!file) {
    return <div className="app-empty">{status || "Loading..."}</div>;
  }

  const imageUrl = sliceUrl(file.name, axis, index);
  const progressPct = propagation
    ? Math.round((100 * propagation.done) / propagation.total)
    : 0;
  const modelsLoading = !pointReady && !runtime?.text_detector_loaded;
  const familyLabel = runtime?.family_label ?? (modelFamily === "sam2" ? "SAM 2" : "SAM 3");
  const textCheckpointOptions =
    runtime?.available_text_checkpoints?.length
      ? runtime.available_text_checkpoints
      : FALLBACK_TEXT_CHECKPOINTS;
  const selectedTextCheckpoint = matchTextCheckpoint(
    textCheckpoint ?? runtime?.text_checkpoint ?? null,
    textCheckpointOptions,
  );
  const loadMessage = runtime?.load_error && !pointReady
    ? runtime.load_error
    : (runtime?.load_stage as string | undefined) ?? "Connecting to inference server...";

  return (
    <div className="app">
      {(modelsLoading || (runtime?.load_error && !pointReady)) && (
        <div className="loading-overlay" role="status">
          <div className="loading-card">
            {!(runtime?.load_error && !pointReady) && <div className="loading-spinner" />}
            <h2>{runtime?.load_error && !pointReady ? "Model failed to load" : `Loading ${familyLabel}`}</h2>
            <p>{loadMessage}</p>
            {!(runtime?.load_error && !pointReady) && (
              <p className="loading-hint">
                Only one tracker stays on the GPU. The first load of a
                model can take a minute (and may download weights).
              </p>
            )}
          </div>
        </div>
      )}
      <aside className="sidebar">
        <h1>Seismic SAM</h1>
        <p className="subtitle">Interactive point segmentation</p>

        <label className="field">
          <span>Tracker model</span>
          <select
            value={modelFamily}
            onChange={(e) => handleModelChange(e.target.value as ModelFamily)}
            disabled={propagation?.running}
            title="Click and volume tracker"
          >
            {(runtime?.available_models ?? FALLBACK_TRACKERS).map((model) => (
              <option key={model.id} value={model.id}>
                {model.label}
              </option>
            ))}
          </select>
          <p className="hint">
            Official click/volume trackers only. Switch to compare masks on
            the same clicks; only one stays on the GPU.
          </p>
        </label>

        {modelFamily === "sam3" && (
          <>
            <label className="field">
              <span>Fine-tuned checkpoint</span>
              <select
                value={selectedTextCheckpoint}
                onChange={(e) => handleTextCheckpointChange(e.target.value)}
                disabled={autoDetecting}
                title="Converted SAM 3 detector weights"
              >
                {textCheckpointOptions.map((item) => (
                  <option key={`${item.source}:${item.path}`} value={item.path}>
                    {item.source === "local" ? `${item.label} (trained)` : item.label}
                  </option>
                ))}
              </select>
              <p className="hint">
                Folders in <code>finetuned_checkpoints/</code> appear here after
                you convert a training run. Detect uses this checkpoint; point
                clicks still use the official SAM 3 tracker.
              </p>
            </label>

            <div className="field">
              <span>Fine-tuned text detector</span>
              <button
                className="primary"
                onClick={handleAutoDetect}
                disabled={!file || autoDetecting}
                title="Run the seismic facies text-prompt model on this slice"
              >
                {autoDetecting ? "Detecting facies..." : "Detect seismic facies"}
              </button>
              {(autoCheckpoint || textCheckpoint) && (
                <p className="hint">
                  Detector: <code>{autoCheckpoint ?? textCheckpoint}</code>
                </p>
              )}
              {autoCoverage !== null && (
                <p className="hint">
                  Facies coverage {(100 * autoCoverage).toFixed(1)}%
                </p>
              )}
              {autoError && <p className="error">{autoError}</p>}
            </div>
          </>
        )}

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
              <option key={f.name} value={f.name}>
                {f.name} — {f.kind.toUpperCase()} {f.shape.join(" × ")}
              </option>
            ))}
          </select>
          {file.format === "npy" && (
            <p className="hint">
              NumPy axes: samples × traces for 2D, or inline × crossline ×
              samples for 3D. VTI export uses 25 m bins and a 4 ms sample
              interval because .npy files do not contain survey headers.
            </p>
          )}
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
              refinement clicks. After a propagate, +/− clicks edit the
              mask on that slice immediately; re-propagate to spread the
              edits through the volume.
            </p>
          )}
          {trackedIds.length > 0 && (
            <p className="hint">
              Tracked volume is live: left/right clicks on any slice edit
              the propagated mask right there.
            </p>
          )}
          {negativeOnlyHere && (
            <p className="hint hint-notice">
              Only − points on this slice and nothing tracked here yet, so
              there is no mask to carve into. Add a + point, or propagate
              first and then refine.
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
              ? `${propagation.resweeping ? "Re-tracking" : "Propagating"} ${
                  propagation.done
                }/${propagation.total}...`
              : !videoReady
                ? "Loading volume tracker..."
                : !needsFullPropagate && pendingEdits > 0
                  ? `Re-propagate ${pendingEdits} slice edit${
                      pendingEdits === 1 ? "" : "s"
                    } (${axis})`
                  : `${trackedIds.length ? "Re-propagate" : "Propagate"} ${
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

        {file.kind === "3d" && canExport && !propagation?.running && (
          <div className="field">
            <span>ParaView export</span>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={includeAmplitude}
                onChange={(e) => setIncludeAmplitude(e.target.checked)}
              />
              Include seismic amplitude
            </label>
            <button
              className="primary"
              onClick={handleExport}
              disabled={exporting}
            >
              {exporting ? "Writing .vti..." : "Download tracked volume (.vti)"}
            </button>
            {pendingEdits > 0 && (
              <p className="hint hint-notice">
                {pendingEdits} slice edit{pendingEdits === 1 ? "" : "s"} not
                swept yet — re-propagate first to carry {pendingEdits === 1 ? "it" : "them"}{" "}
                through the volume.
              </p>
            )}
            {exportInfo && (
              <p className="hint">
                Saved <code>{exportInfo.path}</code> ({exportInfo.size_mb} MB)
                and started a download. Open the .vti in ParaView and
                threshold the <code>label</code> array:{" "}
                {Object.entries(exportInfo.labels)
                  .map(([name, value]) => `${name} = ${value}`)
                  .join(", ")}
                .
              </p>
            )}
            {exportError && <p className="error">{exportError}</p>}
          </div>
        )}

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
              <dt>Tracker</dt>
              <dd>{runtime.family_label}</dd>
              <dt>Checkpoint</dt>
              <dd className="runtime-mono">{runtime.checkpoint}</dd>
              <dt>Facies detector</dt>
              <dd className="runtime-mono">{runtime.text_checkpoint}</dd>
              <dt>Detector loaded</dt>
              <dd>{runtime.text_detector_loaded ? "yes" : "not yet"}</dd>
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
          {segmenting && (
            <div>
              <span className="stat-label">Mask</span>
              <span className="stat-wait">updating...</span>
            </div>
          )}
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
          {autoCoverage !== null && (
            <div>
              <span className="stat-label">Facies coverage</span>
              <span>{(100 * autoCoverage).toFixed(2)}%</span>
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
          autoMaskUrl={autoMaskUrl}
          objects={viewerObjects}
          activeObjectId={activeObjectId}
          sliceWidth={sliceSize.w}
          sliceHeight={sliceSize.h}
          displayWidth={displayWidth}
          displayHeight={displayHeight}
          maskOpacity={maskOpacity}
          busy={
            propagation?.running ||
            autoDetecting ||
            (!pointReady && !runtime?.text_detector_loaded) ||
            (!prepared && trackedIds.length === 0 && !autoMaskUrl)
          }
          onPick={handlePick}
        />
      </main>
    </div>
  );
}
