"""Persistent SAM 3 inference API for the React interactive client.

Run with:
    uvicorn server.main:app --host 127.0.0.1 --port 8000

The models stay resident between requests (mirroring the SAM 2 demo's
client/server split), so a point click only runs the prompt encoder and
mask decoder. Propagation streams one NDJSON event per tracked slice so
the browser can display and scrub frames while the sweep is still
running.
"""

from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from seismic_app import config
from seismic_app.geometry import SectionGeometry
from seismic_app.inference import Sam3PointSegmenter, Sam3VolumePropagator
from seismic_app.logutil import get_logger
from seismic_app.preprocessing import inline_to_rgb_25d, normalize_to_uint8, to_rgb
from seismic_app.sgy_loader import load_any

log = get_logger("server")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

MASK_COLOR = (255, 0, 255)
MASK_ALPHA = 160

app = FastAPI(title="Seismic SAM interactive API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

# One lock serializes GPU work; a second protects the file cache.
_gpu_lock = threading.Lock()
_cache_lock = threading.Lock()
_file_cache: dict[str, tuple[np.ndarray, SectionGeometry, np.ndarray]] = {}
_point_segmenter: Sam3PointSegmenter | None = None
_propagator: Sam3VolumePropagator | None = None


def _list_sgy() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".sgy")


def _get_file(name: str) -> tuple[np.ndarray, SectionGeometry, np.ndarray]:
    allowed = {p.name: p for p in _list_sgy()}
    if name not in allowed:
        raise HTTPException(404, f"Unknown seismic file: {name}")
    with _cache_lock:
        if name not in _file_cache:
            log.info("Loading and normalizing %s...", name)
            data, geometry = load_any(allowed[name])
            _file_cache[name] = (data, geometry, normalize_to_uint8(data))
        return _file_cache[name]


def _get_point_segmenter() -> Sam3PointSegmenter:
    global _point_segmenter
    with _gpu_lock:
        if _point_segmenter is None:
            _point_segmenter = Sam3PointSegmenter(
                config.DEFAULT_CHECKPOINT, embedding_cache_size=4
            )
        return _point_segmenter


def _get_propagator() -> Sam3VolumePropagator:
    global _propagator
    with _gpu_lock:
        if _propagator is None:
            _propagator = Sam3VolumePropagator(config.DEFAULT_CHECKPOINT)
        return _propagator


def _axis_count(data: np.ndarray, geometry: SectionGeometry, axis: str) -> int:
    if geometry.kind == "2d":
        return 1
    return {"inline": data.shape[0], "crossline": data.shape[1], "time": data.shape[2]}[
        axis
    ]


def _slice_rgb(
    data_u8: np.ndarray, geometry: SectionGeometry, axis: str, index: int
) -> np.ndarray:
    if geometry.kind == "2d":
        return to_rgb(data_u8)
    if axis == "inline":
        return inline_to_rgb_25d(data_u8, index)
    if axis == "crossline":
        return to_rgb(data_u8[:, index, :].T)
    return to_rgb(data_u8[:, :, index])


def _validate_slice(
    data: np.ndarray, geometry: SectionGeometry, axis: str, index: int
) -> None:
    if axis not in ("inline", "crossline", "time"):
        raise HTTPException(422, f"Unknown axis: {axis}")
    count = _axis_count(data, geometry, axis)
    if not 0 <= index < count:
        raise HTTPException(422, f"Slice index {index} out of range [0, {count - 1}]")


def _mask_png_base64(mask: np.ndarray) -> str:
    """Encode a boolean mask as a pre-tinted RGBA PNG the browser overlays."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*MASK_COLOR, MASK_ALPHA)
    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG", compress_level=3)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class SliceRef(BaseModel):
    file: str
    axis: str = "inline"
    index: int = 0


class SegmentRequest(SliceRef):
    points: list[list[int]] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)


class PropagateRequest(SliceRef):
    points: list[list[int]] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)


@app.get("/api/files")
def list_files() -> list[dict]:
    entries = []
    for path in _list_sgy():
        data, geometry, _ = _get_file(path.name)
        entries.append(
            {
                "name": path.name,
                "kind": geometry.kind,
                "shape": list(data.shape),
                "axes": {
                    axis: _axis_count(data, geometry, axis)
                    for axis in ("inline", "crossline", "time")
                }
                if geometry.kind == "3d"
                else {"inline": 1, "crossline": 1, "time": 1},
            }
        )
    return entries


@app.get("/api/slice")
def get_slice(file: str, axis: str = "inline", index: int = 0) -> Response:
    data, geometry, data_u8 = _get_file(file)
    _validate_slice(data, geometry, axis, index)
    rgb = _slice_rgb(data_u8, geometry, axis, index)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=90)
    return Response(
        buffer.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=3600"},
    )


@app.post("/api/prepare")
def prepare_slice(req: SliceRef) -> dict:
    """Encode the slice with the vision backbone before the first click."""
    data, geometry, data_u8 = _get_file(req.file)
    _validate_slice(data, geometry, req.axis, req.index)
    rgb = _slice_rgb(data_u8, geometry, req.axis, req.index)
    segmenter = _get_point_segmenter()
    with _gpu_lock:
        segmenter.prepare_image(rgb, image_key=(req.file, req.axis, req.index))
    return {"prepare_seconds": segmenter.last_timings.get("prepare_image", 0.0)}


@app.post("/api/segment")
def segment(req: SegmentRequest) -> dict:
    if len(req.points) != len(req.labels):
        raise HTTPException(422, "points and labels must be equal length")
    data, geometry, data_u8 = _get_file(req.file)
    _validate_slice(data, geometry, req.axis, req.index)
    rgb = _slice_rgb(data_u8, geometry, req.axis, req.index)
    segmenter = _get_point_segmenter()
    started = time.perf_counter()
    with _gpu_lock:
        mask = segmenter.segment(
            rgb,
            [(int(c), int(r)) for c, r in req.points],
            [int(l) for l in req.labels],
            image_key=(req.file, req.axis, req.index),
        )
    return {
        "mask": _mask_png_base64(mask),
        "coverage": float(mask.mean()),
        "timings": {
            **{k: float(v) for k, v in segmenter.last_timings.items()},
            "server_total": time.perf_counter() - started,
        },
    }


@app.post("/api/propagate")
def propagate(req: PropagateRequest) -> StreamingResponse:
    if len(req.points) != len(req.labels):
        raise HTTPException(422, "points and labels must be equal length")
    data, geometry, data_u8 = _get_file(req.file)
    if geometry.kind != "3d":
        raise HTTPException(422, "Propagation requires a 3D volume")
    _validate_slice(data, geometry, req.axis, req.index)
    n_frames = _axis_count(data, geometry, req.axis)
    frames = [_slice_rgb(data_u8, geometry, req.axis, i) for i in range(n_frames)]
    propagator = _get_propagator()

    events: queue.Queue[dict | None] = queue.Queue(maxsize=32)

    def on_progress(done: int, total: int, frame_idx: int, mask: np.ndarray) -> None:
        events.put(
            {
                "type": "frame",
                "frame": int(frame_idx),
                "done": int(done),
                "total": int(total),
                "coverage": float(mask.mean()),
                "mask": _mask_png_base64(mask),
            }
        )

    def run() -> None:
        try:
            with _gpu_lock:
                propagator.propagate(
                    frames,
                    anchor_idx=req.index,
                    points=[(int(c), int(r)) for c, r in req.points],
                    labels=[int(l) for l in req.labels],
                    progress=on_progress,
                )
            events.put(
                {
                    "type": "done",
                    "timings": {
                        k: float(v) for k, v in propagator.last_timings.items()
                    },
                }
            )
        except Exception as exc:  # surface tracker errors to the browser
            log.exception("Propagation failed")
            events.put({"type": "error", "message": str(exc)})
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            event = events.get()
            if event is None:
                break
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
