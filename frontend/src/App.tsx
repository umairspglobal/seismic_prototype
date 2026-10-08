import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react";
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
  getFileStatus,
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
  getFileMeta,
  setModel,
  setTextCheckpoint,
  sliceUrl,
  uploadSeismicFile,
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

const ALL_AXES_READY: Record<Axis, boolean> = { inline: true, crossline: true, time: true };

function formatSeconds(seconds: number | null | undefined): string {
  if (seconds == null) return "?";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
}

const FALLBACK_TRACKERS = [
  { id: "sam3" as const, label: "SAM 3", checkpoint: "facebook/sam3", gated: true },
  { id: "sam31" as const, label: "SAM 3.1 (In Progress)", checkpoint: "facebook/sam3.1", gated: true },
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

function isSeismicUpload(name: string): boolean {
  const lower = name.toLowerCase();
  return lower.endsWith(".sgy") || lower.endsWith(".npy");
}

function SurveyUpload({
  busy,
  percent,
  currentName,
  error,
  inputRef,
  onPick,
}: {
  busy: boolean;
  percent: number | null;
  currentName: string | null;
  error: string | null;
  inputRef: RefObject<HTMLInputElement>;
  onPick: (files: FileList | null) => void;
}) {
  const label = busy
    ? percent != null && percent < 100
      ? `Uploading ${currentName ?? "file"} ${percent}%`
      : `Saving ${currentName ?? "file"} to data/…`
    : "Upload .sgy or .npy";
  return (
    <div className="field">
      <span>Upload a survey</span>
      <button
        type="button"
        className="upload-button"
        disabled={busy}
        onClick={() => inputRef.current?.click()}
      >
        {label}
      </button>
      <input
        ref={inputRef}
        className="file-input"
        type="file"
        accept=".sgy,.npy"
        multiple
        onChange={(event) => {
          onPick(event.target.files);
          event.target.value = "";
        }}
      />
      {busy && (
        <div className="progress-wrap" aria-hidden="true">
          <div className="progress-bar" style={{ width: `${percent ?? 0}%` }} />
        </div>
      )}
      {error && <p className="error">{error}</p>}
      <p className="hint">
        The file is copied into <code>data/</code> and opened in the viewer.
      </p>
    </div>
  );
}

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
  const [sliceError, setSliceError] = useState<string | null>(null);
  const [sliceLoading, setSliceLoading] = useState(false);
  const sliceRequestedAt = useRef(0);
  // Propagation half-width as typed; empty means the server picks (the
  // whole axis for small files, a RAM-sized window for large ones).
  const [propWindow, setPropWindow] = useState("");
  const [trackedRange, setTrackedRange] = useState<{
    start: number;
    stop: number;
    liveStart: number;
    liveStop: number;
  } | null>(null);
  const lastStageRef = useRef<string | null>(null);
  const userPickedTextCheckpoint = useRef(false);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const [uploadName, setUploadName] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
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

  const applyFiles = useCallback((incoming: FileInfo[]) => {
    if (!incoming.length) return;
    setFiles((prev) => {
      const byName = new Map(prev.map((item) => [item.name, item]));
      for (const item of incoming) byName.set(item.name, item);
      return Array.from(byName.values()).sort((a, b) =>
        a.name.localeCompare(b.name, undefined, { sensitivity: "base" }),
      );
    });
    setFile((current) => {
      const updated = incoming.find((item) => item.name === current?.name);
      if (updated) return updated;
      if (current) return current;
      return incoming.find((item) => item.kind === "3d") ?? incoming[0] ?? null;
    });
    setStatus("");
  }, []);

  const handleUpload = useCallback(
    async (list: FileList | null) => {
      const picked = Array.from(list ?? []);
      if (!picked.length || uploading) return;
      const rejected = picked.find((item) => !isSeismicUpload(item.name));
      if (rejected) {
        setUploadError(`${rejected.name} is not a .sgy or .npy file.`);
        return;
      }
      setUploading(true);
      setUploadError(null);
      try {
        for (const item of picked) {
          setUploadName(item.name);
          setUploadPercent(0);
          const saved = await uploadSeismicFile(item, setUploadPercent);
          setUploadPercent(100);
          applyFiles([saved]);
          setFile(saved);
          setIndex(0);
          propMasksRef.current = new Map();
          setPropagation(null);
          setSliceError(null);
          setStatus("");
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Upload failed";
        setUploadError(message);
      } finally {
        setUploading(false);
        setUploadPercent(null);
        setUploadName(null);
      }
    },
    [applyFiles, uploading],
  );

  useEffect(() => {
    let cancelled = false;
    let sawAny = false;
    let sawError = false;

    // NPY metadata is instant. SEG-Y listing is separate so a huge volume
    // cannot hide the rest of the dropdown while its 3D headers are indexed.
    const load = (format: "npy" | "sgy") =>
      listFiles(format)
        .then((incoming) => {
          if (cancelled) return;
          if (incoming.length) {
            sawAny = true;
            applyFiles(incoming);
          }
        })
        .catch(() => {
          sawError = true;
        });

    Promise.all([load("npy"), load("sgy")]).then(() => {
      if (cancelled) return;
      if (!sawAny && sawError) {
        setStatus(
          "Cannot reach the inference server. Start it with: uvicorn server.main:app",
        );
      } else if (!sawAny) {
        setStatus("No surveys in data/ yet. Upload a .sgy or .npy file to open it.");
      }
    });
    return () => {
      cancelled = true;
    };
  }, [applyFiles]);

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

  // Wide 2D lines are served as pages of traces; the page is the "slice".
  const pageCount = file?.kind === "2d" ? (file.pages?.count ?? 1) : 1;
  const axisCount = file ? (file.kind === "2d" ? pageCount : file.axes[axis]) : 1;
  const axesReady = file?.large ? (file.axes_ready ?? { inline: false, crossline: false, time: false }) : ALL_AXES_READY;
  const axisReady = file?.kind === "2d" ? axesReady.inline : axesReady[axis];
  const cacheStatus = file?.large ? (file.status ?? null) : null;
  const sliceSize = useMemo(() => {
    if (!file) return { w: 1, h: 1 };
    const [nIl, nXl, nS] = file.shape;
    // 2D lines are stored (n_samples, n_traces): time down, traces across.
    if (file.kind === "2d") return { w: file.pages?.width ?? file.shape[1], h: file.shape[0] };
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
    setTrackedRange(null);
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
      setStatus("");
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
  }, [file?.name, axis]);

  // Opening a survey is independent of the tracker. Refresh kind/shape
  // after the server actually loads the file (listing may have used a
  // fast 2D header fallback for a slow 3D volume).
  useEffect(() => {
    if (!file) return;
    let cancelled = false;
    setSliceError(null);
    const openedAt = performance.now();
    console.info(`[seismic] opening ${file.name}...`);
    getFileMeta(file.name)
      .then((info) => {
        if (cancelled) return;
        console.info(
          `[seismic] opened ${info.name}: ${info.kind.toUpperCase()} ${info.shape.join("x")}` +
            (info.large
              ? `, large file (cache ${info.status?.stage ?? "?"} ${info.status?.percent ?? 0}%)`
              : ", in memory") +
            ` in ${Math.round(performance.now() - openedAt)} ms`,
        );
        setFile((current) => {
          if (!current || current.name !== info.name) return current;
          const unchanged =
            current.kind === info.kind &&
            current.shape.length === info.shape.length &&
            current.shape.every((size, i) => size === info.shape[i]) &&
            !info.large &&
            !current.large &&
            (current.pages?.count ?? 1) === (info.pages?.count ?? 1);
          return unchanged ? current : { ...current, ...info };
        });
      })
      .catch((err: Error) => {
        if (cancelled) return;
        const message = err.message || "";
        if (
          /Could not load seismic file|Unknown seismic file/i.test(message)
        ) {
          setSliceError(message);
          setStatus(`Could not open ${file.name}: ${message}`);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [file?.name]);

  // Large files: poll the background cache build until it finishes, so
  // the progress overlay and the axis gating stay current.
  const cacheSettled = !file?.large || Boolean(file.status?.ready) || file.status?.stage === "error";
  useEffect(() => {
    if (!file?.large || cacheSettled) return;
    let cancelled = false;
    let timer: number | undefined;
    const poll = () => {
      timer = window.setTimeout(async () => {
        try {
          const info = await getFileStatus(file.name);
          if (cancelled) return;
          setFile((current) =>
            current && current.name === info.name ? { ...current, ...info } : current,
          );
          setFiles((prev) => prev.map((f) => (f.name === info.name ? { ...f, ...info } : f)));
          if (!info.status?.ready && info.status?.stage !== "error") poll();
        } catch {
          if (!cancelled) poll();
        }
      }, 1000);
    };
    poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [file?.name, file?.large, cacheSettled]);

  useEffect(() => {
    if (!file || !cacheStatus) return;
    const key = `${file.name}:${cacheStatus.stage}`;
    if (lastStageRef.current === key) return;
    lastStageRef.current = key;
    console.info(
      `[seismic] ${file.name} cache: ${cacheStatus.stage} (${cacheStatus.percent}%) - ${cacheStatus.message}`,
    );
  }, [file, cacheStatus]);

  // An axis of a large file that is still being cached cannot be shown;
  // fall back to one that can (usually inline) instead of a blank viewer.
  useEffect(() => {
    if (!file?.large || file.kind !== "3d" || axisReady) return;
    const ready = (["inline", "crossline", "time"] as Axis[]).find((a) => axesReady[a]);
    if (ready && ready !== axis) {
      console.info(`[seismic] ${axis} axis not cached yet; switching to ${ready}`);
      setAxis(ready);
      setIndex(0);
    }
  }, [file?.name, file?.large, file?.kind, axis, axisReady, axesReady]);

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
            "Facies detector needed the GPU, so the click tracker was unloaded. Re-select a tracker model to reload it.",
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
    if (!file || !pointReady || !axisReady) return;
    abortAll();
    setSegmenting(false);
    setObjects((prev) => prev.map((o) => ({ ...o, maskUrl: null })));
    setPropMaskUrl(null);
    const propagated = propMasksRef.current.get(index);
    if (propagation && propagation.axis === axis && propagated) {
      setPropMaskUrl(maskDataUrl(propagated));
    }
    setPrepared(false);
    setSliceError(null);
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
        .catch((err: Error) => {
          setPrepared(false);
          setSliceError(`Could not prepare this slice: ${err.message}`);
        });
    }, 250);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file?.name, axis, index, pointReady, axisReady]);

  const imageUrl = file && axisReady ? sliceUrl(file.name, axis, index) : null;
  useEffect(() => {
    if (!imageUrl) {
      setSliceLoading(false);
      return;
    }
    sliceRequestedAt.current = performance.now();
    setSliceLoading(true);
  }, [imageUrl]);

  const handleSliceLoad = useCallback(() => {
    setSliceLoading(false);
    setSliceError(null);
    console.info(
      `[seismic] slice ${file?.name} ${axis}[${index}] loaded in ${Math.round(
        performance.now() - sliceRequestedAt.current,
      )} ms`,
    );
  }, [file?.name, axis, index]);

  const handleSliceError = useCallback(() => {
    setSliceLoading(false);
    console.warn(`[seismic] slice ${file?.name} ${axis}[${index}] failed to load`);
    setSliceError(
      file?.large && !axisReady
        ? "This axis is still being cached."
        : "Could not load this slice from the inference server.",
    );
  }, [file?.name, file?.large, axis, index, axisReady]);

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
    const requestedWindow = propWindow.trim() === "" ? null : Math.max(0, Math.floor(Number(propWindow)));
    const windowArg = requestedWindow != null && Number.isFinite(requestedWindow) ? requestedWindow : null;
    setPropagation({
      axis,
      running: true,
      done: 0,
      total: windowArg != null ? Math.min(axisCount, 2 * windowArg + 1) : axisCount,
      resweeping: reuseSessions,
    });
    setExportInfo(null);
    const propagateStarted = performance.now();
    console.info(
      `[seismic] ${reuseSessions ? "re-sweep" : "propagate"} ${file.name} ${axis} from slice ${index}` +
        ` (range ${windowArg != null ? `+/-${windowArg}` : "auto"})`,
    );

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
        if (event.start != null && event.stop != null) {
          const { start, stop } = event;
          setTrackedRange((prev) => ({
            start,
            stop,
            // A re-sweep reports only the live range; keep the full extent.
            liveStart: event.live_start ?? prev?.liveStart ?? start,
            liveStop: event.live_stop ?? prev?.liveStop ?? stop,
            ...(reuseSessions && prev ? { start: prev.start, stop: prev.stop } : {}),
          }));
        }
        console.info(
          `[seismic] tracking done in ${((performance.now() - propagateStarted) / 1000).toFixed(1)} s` +
            (event.start != null ? `, slices ${event.start}-${(event.stop ?? 0) - 1}` : "") +
            (event.window_reason ? ` (${event.window_reason})` : ""),
        );
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
      : propagate(file.name, axis, index, objectsWithPoints, onEvent, undefined, windowArg);
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
    propWindow,
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

  const uploadControl = (
    <SurveyUpload
      busy={uploading}
      percent={uploadPercent}
      currentName={uploadName}
      error={uploadError}
      inputRef={uploadInputRef}
      onPick={(picked) => {
        void handleUpload(picked);
      }}
    />
  );

  if (!file) {
    return (
      <div className="app-empty">
        <div className="stage-status">
          <h2>Seismic SAM</h2>
          <p>{status || "Loading surveys..."}</p>
          {uploadControl}
        </div>
      </div>
    );
  }

  const pageStart =
    file.kind === "2d" && file.pages
      ? Math.min(index * file.pages.step, file.shape[1] - file.pages.width)
      : 0;
  const progressPct = propagation
    ? Math.round((100 * propagation.done) / propagation.total)
    : 0;
  const detectorCanStayInteractive =
    modelFamily === "sam3" && Boolean(runtime?.text_detector_loaded);
  const modelsLoading = !pointReady && !detectorCanStayInteractive;
  const familyLabel =
    runtime?.family_label ??
    (modelFamily === "sam2"
      ? "SAM 2"
      : modelFamily === "sam31"
        ? "SAM 3.1 (In Progress)"
        : "SAM 3");
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
  const trackerBlocked = Boolean(modelsLoading || (runtime?.load_error && !pointReady));

  return (
    <div className="app">
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
            {modelFamily === "sam31"
              ? "Object Multiplex tracks objects in shared memory. It requires the latest native SAM 3 code and CUDA. You can still open another seismic file while it loads."
              : "Official click/volume trackers only. Switch to compare masks on the same clicks; only one stays on the GPU. Seismic files stay available while a tracker loads."}
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
                setSliceError(null);
                setStatus("");
              }
            }}
          >
            {files.map((f) => (
              <option key={f.name} value={f.name}>
                {f.name} — {f.kind.toUpperCase()} {f.shape.join(" × ")}
                {f.large ? (f.status?.ready ? " (large, cached)" : " (large)") : ""}
              </option>
            ))}
          </select>
          {file.large && (
            <p className="hint">
              Large file — served from a disk cache instead of RAM
              {cacheStatus?.ready
                ? "; all axes are ready."
                : cacheStatus?.stage === "error"
                  ? "; the cache build failed (see below)."
                  : `; building the cache (${cacheStatus?.percent ?? 0}%).`}
            </p>
          )}
          {trackerBlocked && (
            <p className="hint hint-notice">
              {runtime?.load_error && !pointReady
                ? "Tracker failed to load — switch models above. You can still open another seismic file."
                : `Loading ${familyLabel} — you can still open another .sgy or .npy file. Clicks start when it is ready.`}
            </p>
          )}
          {file.format === "npy" && (
            <p className="hint">
              NumPy axes: samples × traces for 2D, or inline × crossline ×
              samples for 3D. VTI export uses 25 m bins and a 4 ms sample
              interval because .npy files do not contain survey headers.
            </p>
          )}
        </label>

        {uploadControl}

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
                {(
                  [
                    ["inline", "Inline"],
                    ["crossline", "Crossline"],
                    ["time", "Time slice"],
                  ] as [Axis, string][]
                ).map(([value, label]) => (
                  <option
                    key={value}
                    value={value}
                    disabled={!axesReady[value]}
                    title={axesReady[value] ? undefined : "Available when the cache build finishes"}
                  >
                    {axesReady[value]
                      ? label
                      : `${label} (building cache ${cacheStatus?.percent ?? 0}%)`}
                  </option>
                ))}
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
              {trackedRange && propagation?.axis === axis && axisCount > 1 && (
                <div
                  className="range-track"
                  title={`Tracked slices ${trackedRange.start + 1}-${trackedRange.stop}`}
                >
                  <div
                    className="range-track-fill"
                    style={{
                      left: `${(100 * trackedRange.start) / axisCount}%`,
                      width: `${(100 * (trackedRange.stop - trackedRange.start)) / axisCount}%`,
                    }}
                  />
                </div>
              )}
            </label>
          </>
        )}

        {file.kind === "2d" && pageCount > 1 && (
          <label className="field">
            <span>
              Trace page {index + 1} of {pageCount}
              <em className="hint">
                {" "}
                — traces {pageStart + 1}-{pageStart + sliceSize.w}
              </em>
            </span>
            <input
              type="range"
              min={0}
              max={pageCount - 1}
              value={index}
              onChange={(e) => setIndex(Number(e.target.value))}
            />
          </label>
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
          <label className="field">
            <span>Propagation range ± slices</span>
            <input
              className="number-input"
              type="number"
              min={1}
              placeholder="auto (whole axis)"
              value={propWindow}
              disabled={propagation?.running}
              onChange={(e) => setPropWindow(e.target.value)}
            />
            {trackedRange && propagation?.axis === axis && (
              <p className="hint">
                Tracked slices {trackedRange.start + 1}-{trackedRange.stop} of {axisCount}.
                {(trackedRange.liveStart > trackedRange.start ||
                  trackedRange.liveStop < trackedRange.stop) &&
                  ` Clicks edit the tracked mask on slices ${trackedRange.liveStart + 1}-${trackedRange.liveStop}; elsewhere they preview that slice only (re-propagate from there to track the change).`}
                {(trackedRange.start > 0 || trackedRange.stop < axisCount) &&
                  " Slices outside the tracked range need a wider propagation."}
              </p>
            )}
          </label>
        )}

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
        {trackerBlocked && (
          <div
            className={`stage-status ${
              runtime?.load_error && !pointReady ? "stage-status-error" : ""
            }`}
            role="status"
          >
            {!(runtime?.load_error && !pointReady) && (
              <div className="loading-spinner" />
            )}
            <h2>
              {runtime?.load_error && !pointReady
                ? "Tracker failed to load"
                : `Loading ${familyLabel}`}
            </h2>
            <p>{loadMessage}</p>
            <p className="loading-hint">
              Seismic files stay available in the sidebar. Clicks wait until
              the tracker is ready. Only one tracker stays on the GPU.
            </p>
          </div>
        )}
        {cacheStatus && !cacheStatus.ready && (
          <div
            className={`stage-status ${cacheStatus.stage === "error" ? "stage-status-error" : ""}`}
            role="status"
          >
            {cacheStatus.stage !== "error" && <div className="loading-spinner" />}
            <h2>
              {cacheStatus.stage === "error"
                ? `Could not prepare ${file.name}`
                : `Preparing ${file.name}`}
            </h2>
            <p>{cacheStatus.error ?? cacheStatus.message}</p>
            {cacheStatus.stage !== "error" && (
              <>
                <div className="progress-wrap cache-progress">
                  <div className="progress-bar" style={{ width: `${cacheStatus.percent}%` }} />
                </div>
                <p className="loading-hint">
                  {cacheStatus.stage} · {cacheStatus.percent}%
                  {cacheStatus.mb_per_s != null && ` · ${Math.round(cacheStatus.mb_per_s)} MB/s`}
                  {cacheStatus.eta_s != null && ` · about ${formatSeconds(cacheStatus.eta_s)} left`}
                </p>
                <p className="loading-hint">
                  {axisReady
                    ? "You can already view and segment this axis. The other axes unlock when the one-time cache is finished; later opens are instant."
                    : "This one-time cache makes every axis load instantly next time."}
                </p>
              </>
            )}
          </div>
        )}
        {sliceError && <p className="stage-error">{sliceError}</p>}
        {imageUrl && (
          <div className="viewer-wrap">
            {sliceLoading && (
              <div className="slice-loading" role="status">
                <div className="loading-spinner" />
              </div>
            )}
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
                (!pointReady && !detectorCanStayInteractive) ||
                (!prepared && trackedIds.length === 0 && !autoMaskUrl && !sliceError)
              }
              onPick={handlePick}
              onImageLoad={handleSliceLoad}
              onImageError={handleSliceError}
            />
          </div>
        )}
      </main>
    </div>
  );
}
