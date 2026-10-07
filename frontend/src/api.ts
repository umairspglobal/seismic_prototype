export type Axis = "inline" | "crossline" | "time";

/** Background cache build of a large file (see /api/file-status). */
export interface CacheStatus {
  stage: "queued" | "scanning" | "sampling" | "converting" | "finalizing" | "ready" | "error";
  message: string;
  percent: number;
  mb_per_s: number | null;
  eta_s: number | null;
  elapsed_s: number | null;
  error: string | null;
  ready: boolean;
  cache_dir: string;
}

/** Overlapping trace windows of a 2D line too wide to show at once. */
export interface PageInfo {
  count: number;
  width: number;
  step: number;
}

export interface FileInfo {
  name: string;
  format: "sgy" | "npy";
  kind: "2d" | "3d";
  shape: number[];
  axes: { inline: number; crossline: number; time: number };
  /** Served from the disk cache instead of being loaded into RAM. */
  large?: boolean;
  pages?: PageInfo | null;
  /** Large files only: axes whose slices can be served right now. */
  axes_ready?: Record<Axis, boolean>;
  status?: CacheStatus | null;
}

export type ModelFamily = "sam2" | "sam3" | "sam31";

export interface ModelInfo {
  id: ModelFamily;
  label: string;
  checkpoint: string;
  gated: boolean;
}

export interface TextCheckpointInfo {
  id: string;
  label: string;
  path: string;
  source: "official" | "local" | "env";
}

export interface Point {
  col: number;
  row: number;
  label: 0 | 1;
  /** Slice index the point was picked on (0 for 2D lines). */
  slice: number;
}

export interface ObjectPrompt {
  id: number;
  points: Point[];
}

// Must stay in sync with OBJECT_COLORS in server/main.py (mask tints).
export const OBJECT_COLORS = [
  "#ff00ff", // magenta
  "#00bfff", // sky blue
  "#ffd600", // yellow
  "#19e083", // mint
  "#ff6d00", // orange
  "#a26bff", // violet
];

export function objectColor(objectId: number): string {
  return OBJECT_COLORS[objectId % OBJECT_COLORS.length];
}

export interface SegmentResult {
  mask: string;
  coverage: number;
  timings: Record<string, number>;
}

export interface FrameEvent {
  type: "frame";
  frame: number;
  done: number;
  total: number;
  coverage: number;
  mask: string;
}

export interface DoneEvent {
  type: "done";
  timings: Record<string, number>;
  /** Tracker sessions were kept, so slices can be edited in place. */
  editable?: boolean;
  /** Tracked slice range [start, stop) along the axis. */
  start?: number;
  stop?: number;
  /** Slices held by the live tracker session, where clicks refine the mask. */
  live_start?: number;
  live_stop?: number;
  window_reason?: string;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type PropagationEvent = FrameEvent | DoneEvent | ErrorEvent;

/** A propagated volume the server still holds, editable and exportable. */
export interface TrackedVolume {
  file: string;
  axis: Axis;
  objects: number[];
  edited_slices: number[];
  start?: number;
  stop?: number;
}

export interface ExportResult {
  path: string;
  directory: string;
  size_mb: number;
  /** Object name -> integer value in the exported "label" array. */
  labels: Record<string, number>;
  seconds: number;
}

export interface RuntimeInfo {
  family: ModelFamily;
  family_label: string;
  available_models: ModelInfo[];
  checkpoint: string;
  architecture: string;
  point_model: string;
  video_model: string;
  point_loaded: boolean;
  video_loaded: boolean;
  load_stage: string | null;
  load_error: string | null;
  ready: boolean;
  point_device: string | null;
  video_device: string | null;
  video_precision: string | null;
  embedding_cache_size: number | null;
  cached_slices: number;
  tracked_volume: TrackedVolume | null;
  hardware: {
    cuda_available: boolean;
    device_name: string;
    compute_capability: string | null;
    cuda_version: string | null;
    gpu_count: number;
    vram: {
      allocated_gb: number | null;
      reserved_gb: number | null;
      total_gb: number | null;
    };
  };
  software: {
    python: string;
    platform: string;
    torch: string;
    transformers: string | null;
  };
  text_checkpoint: string;
  available_text_checkpoints: TextCheckpointInfo[];
  text_detector_loaded: boolean;
}

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8000";

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

/** FastAPI reports failures as {"detail": ...}; surface that, not the code. */
async function errorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (detail) return JSON.stringify(detail);
  } catch {
    /* not JSON */
  }
  return `${fallback} (${res.status})`;
}

async function postJson<T>(
  path: string,
  body: unknown,
  fallback: string,
  signal?: AbortSignal,
): Promise<T> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await errorMessage(res, fallback));
  return res.json();
}

/** Read an NDJSON stream of tracking events, dispatching one per line. */
async function readEventStream(
  res: Response,
  onEvent: (event: PropagationEvent) => void,
): Promise<void> {
  if (!res.body) throw new Error("The server returned an empty stream");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line) as PropagationEvent);
    }
  }
}

export async function listFiles(format?: "npy" | "sgy"): Promise<FileInfo[]> {
  const query = format ? `?format=${format}` : "";
  const res = await fetch(apiUrl(`/api/files${query}`));
  if (!res.ok) throw new Error(`Failed to list files: ${res.status}`);
  return res.json();
}

export async function getFileMeta(file: string): Promise<FileInfo> {
  const params = new URLSearchParams({ file });
  const res = await fetch(apiUrl(`/api/meta?${params}`));
  if (!res.ok) throw new Error(await errorMessage(res, "Failed to open seismic file"));
  return res.json();
}

/** Cache progress and axis readiness; cheap enough to poll every second. */
export async function getFileStatus(file: string): Promise<FileInfo> {
  const params = new URLSearchParams({ file });
  const res = await fetch(apiUrl(`/api/file-status?${params}`));
  if (!res.ok) throw new Error(await errorMessage(res, "Failed to read file status"));
  return res.json();
}

export async function getRuntime(): Promise<RuntimeInfo> {
  const res = await fetch(apiUrl("/api/runtime"));
  if (!res.ok) throw new Error(`Failed to load runtime info: ${res.status}`);
  return res.json();
}

export async function setModel(
  family: ModelFamily,
): Promise<{ family: ModelFamily; ready: boolean; label: string }> {
  return postJson(
    "/api/model",
    { family },
    "Failed to switch tracker model",
  );
}

export async function setTextCheckpoint(
  checkpoint: string,
): Promise<{ checkpoint: string; label: string; loaded: boolean }> {
  return postJson(
    "/api/text-checkpoint",
    { checkpoint },
    "Failed to switch fine-tuned checkpoint",
  );
}

export function sliceUrl(file: string, axis: Axis, index: number): string {
  const params = new URLSearchParams({ file, axis, index: String(index) });
  return apiUrl(`/api/slice?${params}`);
}

export async function prepareSlice(
  file: string,
  axis: Axis,
  index: number,
): Promise<void> {
  const res = await fetch(apiUrl("/api/prepare"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file, axis, index }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Prepare failed: ${res.status}`);
  }
}

export async function autoSegment(
  file: string,
  axis: Axis,
  index: number,
  signal?: AbortSignal,
): Promise<SegmentResult & { prompt: string; checkpoint: string; unloaded_tracker: boolean }> {
  return postJson(
    "/api/auto-segment",
    { file, axis, index },
    "Facies detection failed",
    signal,
  );
}

export async function segment(
  file: string,
  axis: Axis,
  index: number,
  points: Point[],
  objectId: number,
  signal?: AbortSignal,
): Promise<SegmentResult> {
  return postJson<SegmentResult>(
    "/api/segment",
    {
      file,
      axis,
      index,
      points: points.map((p) => [p.col, p.row]),
      labels: points.map((p) => p.label),
      object_id: objectId,
    },
    "Segmentation failed",
    signal,
  );
}

/**
 * Edit one object on one slice of an already-tracked volume.
 *
 * Unlike `segment`, this runs against the live tracker session, so the
 * clicks are combined with the tracker's memory of the object on this
 * slice: negative points carve into the propagated mask instead of
 * being meaningless on their own. Returns the slice's combined
 * multi-object overlay.
 */
export async function refine(
  file: string,
  axis: Axis,
  index: number,
  points: Point[],
  objectId: number,
  signal?: AbortSignal,
): Promise<SegmentResult> {
  return postJson<SegmentResult>(
    "/api/refine",
    {
      file,
      axis,
      index,
      points: points.map((p) => [p.col, p.row]),
      labels: points.map((p) => p.label),
      object_id: objectId,
    },
    "Refinement failed",
    signal,
  );
}

export async function propagate(
  file: string,
  axis: Axis,
  anchor: number,
  objects: ObjectPrompt[],
  onEvent: (event: PropagationEvent) => void,
  signal?: AbortSignal,
  /** Half-width of the tracked range; null lets the server choose. */
  window: number | null = null,
): Promise<void> {
  const res = await fetch(apiUrl("/api/propagate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      file,
      axis,
      index: anchor,
      ...(window != null ? { window } : {}),
      objects: objects.map((o) => ({
        id: o.id,
        points: o.points.map((p) => [p.col, p.row]),
        labels: o.points.map((p) => p.label),
        slices: o.points.map((p) => p.slice),
      })),
    }),
  });
  if (!res.ok) throw new Error(await errorMessage(res, "Propagation failed"));
  await readEventStream(res, onEvent);
}

/**
 * Re-track the volume outward from the slices edited since the last sweep.
 *
 * Reuses the live tracker sessions, so it is much faster than a full
 * propagation and keeps every refinement click already applied.
 */
export async function resweep(
  file: string,
  axis: Axis,
  onEvent: (event: PropagationEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(apiUrl("/api/resweep"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({ file, axis }),
  });
  if (!res.ok) throw new Error(await errorMessage(res, "Re-propagation failed"));
  await readEventStream(res, onEvent);
}

/** Write the tracked volume to a ParaView .vti in the outputs/ folder. */
export async function exportVolume(
  file: string,
  axis: Axis,
  includeAmplitude: boolean,
): Promise<ExportResult> {
  return postJson<ExportResult>(
    "/api/export",
    { file, axis, include_amplitude: includeAmplitude },
    "Export failed",
  );
}

/** Download a previously written .vti so it can be opened in ParaView. */
export async function downloadExportedVolume(
  file: string,
  axis: Axis,
  filename: string,
): Promise<void> {
  const params = new URLSearchParams({ file, axis });
  const res = await fetch(apiUrl(`/api/export/file?${params}`));
  if (!res.ok) throw new Error(await errorMessage(res, "Download failed"));
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function maskDataUrl(base64Png: string): string {
  return `data:image/png;base64,${base64Png}`;
}
