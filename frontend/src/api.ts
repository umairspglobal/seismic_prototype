export interface FileInfo {
  name: string;
  kind: "2d" | "3d";
  shape: number[];
  axes: { inline: number; crossline: number; time: number };
}

export type Axis = "inline" | "crossline" | "time";

export interface Point {
  col: number;
  row: number;
  label: 0 | 1;
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
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type PropagationEvent = FrameEvent | DoneEvent | ErrorEvent;

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8000";

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export async function listFiles(): Promise<FileInfo[]> {
  const res = await fetch(apiUrl("/api/files"));
  if (!res.ok) throw new Error(`Failed to list files: ${res.status}`);
  return res.json();
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
  await fetch(apiUrl("/api/prepare"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file, axis, index }),
  });
}

export async function segment(
  file: string,
  axis: Axis,
  index: number,
  points: Point[],
  signal?: AbortSignal,
): Promise<SegmentResult> {
  const res = await fetch(apiUrl("/api/segment"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      file,
      axis,
      index,
      points: points.map((p) => [p.col, p.row]),
      labels: points.map((p) => p.label),
    }),
  });
  if (!res.ok) throw new Error(`Segmentation failed: ${res.status}`);
  return res.json();
}

export async function propagate(
  file: string,
  axis: Axis,
  anchor: number,
  points: Point[],
  onEvent: (event: PropagationEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(apiUrl("/api/propagate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      file,
      axis,
      index: anchor,
      points: points.map((p) => [p.col, p.row]),
      labels: points.map((p) => p.label),
    }),
  });
  if (!res.ok || !res.body) throw new Error(`Propagation failed: ${res.status}`);

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

export function maskDataUrl(base64Png: string): string {
  return `data:image/png;base64,${base64Png}`;
}
