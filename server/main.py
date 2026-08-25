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
import platform
import queue
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from seismic_app import config
from seismic_app.geometry import SectionGeometry
from seismic_app.inference import (
    Sam3PointSegmenter,
    Sam3VolumePropagator,
    transformers_version,
)
from seismic_app.logutil import get_logger
from seismic_app.preprocessing import inline_to_rgb_25d, normalize_to_uint8, to_rgb
from seismic_app.sgy_loader import load_any

log = get_logger("server")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

MASK_ALPHA = 160
# Per-object tint palette; must stay in sync with OBJECT_COLORS in
# frontend/src/api.ts (markers and sidebar swatches use the same colors).
OBJECT_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 255),  # magenta
    (0, 191, 255),  # sky blue
    (255, 214, 0),  # yellow
    (25, 224, 131),  # mint
    (255, 109, 0),  # orange
    (162, 107, 255),  # violet
]


def _object_color(object_id: int) -> tuple[int, int, int]:
    return OBJECT_COLORS[object_id % len(OBJECT_COLORS)]


# One lock serializes GPU work; a second protects the file cache.
_gpu_lock = threading.Lock()
_cache_lock = threading.Lock()
_file_cache: dict[str, tuple[np.ndarray, SectionGeometry, np.ndarray]] = {}
_point_segmenter: Sam3PointSegmenter | None = None
_propagator: Sam3VolumePropagator | None = None
_load_state: dict[str, str | bool | None] = {
    "stage": "starting",
    "error": None,
}


def _warmup_models() -> None:
    """Load both trackers at process start so the first click is not a cold load."""
    try:
        _load_state["stage"] = "Loading SAM 3 point tracker onto the GPU..."
        _get_point_segmenter()
        _load_state["stage"] = "Loading SAM 3 volume tracker onto the GPU..."
        _get_propagator()
        _load_state["stage"] = "ready"
        log.info("Model warmup complete; point and volume trackers are resident.")
    except Exception as exc:
        log.exception("Model warmup failed")
        _load_state["error"] = str(exc)
        _load_state["stage"] = "ready" if _point_segmenter is not None else "error"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warmup_models, daemon=True, name="model-warmup").start()
    yield


app = FastAPI(title="Seismic SAM interactive API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _require_point_segmenter() -> Sam3PointSegmenter:
    if _load_state["error"] and _point_segmenter is None:
        raise HTTPException(503, f"Model failed to load: {_load_state['error']}")
    if _point_segmenter is None:
        raise HTTPException(
            503, str(_load_state["stage"] or "SAM 3 point tracker is still loading")
        )
    return _point_segmenter


def _require_propagator() -> Sam3VolumePropagator:
    if _load_state["error"] and _propagator is None:
        raise HTTPException(503, f"Model failed to load: {_load_state['error']}")
    if _propagator is None:
        raise HTTPException(
            503, str(_load_state["stage"] or "SAM 3 volume tracker is still loading")
        )
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


def _rgba_png_base64(rgba: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG", compress_level=3)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _mask_png_base64(mask: np.ndarray, object_id: int = 0) -> str:
    """Encode a boolean mask as a pre-tinted RGBA PNG the browser overlays."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*_object_color(object_id), MASK_ALPHA)
    return _rgba_png_base64(rgba)


def _masks_png_base64(masks: np.ndarray, object_ids: list[int]) -> str:
    """Encode a (n_objects, H, W) bool stack as one combined tinted PNG."""
    rgba = np.zeros((*masks.shape[1:], 4), dtype=np.uint8)
    for mask, object_id in zip(masks, object_ids):
        rgba[mask] = (*_object_color(object_id), MASK_ALPHA)
    return _rgba_png_base64(rgba)


class SliceRef(BaseModel):
    file: str
    axis: str = "inline"
    index: int = 0


class SegmentRequest(SliceRef):
    points: list[list[int]] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)
    object_id: int = 0  # selects the overlay tint color


class ObjectPrompt(BaseModel):
    id: int = 0
    points: list[list[int]] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)


class PropagateRequest(SliceRef):
    objects: list[ObjectPrompt] = Field(min_length=1)


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


def _bytes_to_gb(n: int | None) -> float | None:
    if n is None:
        return None
    return round(n / (1024**3), 2)


@app.get("/api/runtime")
def runtime_info() -> dict:
    """Hardware, library, checkpoint, and loaded-model status for the sidebar."""
    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else None
    capability = None
    vram: dict[str, float | None] = {
        "allocated_gb": None,
        "reserved_gb": None,
        "total_gb": None,
    }
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        capability = f"{props.major}.{props.minor}"
        vram = {
            "allocated_gb": _bytes_to_gb(torch.cuda.memory_allocated(0)),
            "reserved_gb": _bytes_to_gb(torch.cuda.memory_reserved(0)),
            "total_gb": _bytes_to_gb(props.total_memory),
        }

    try:
        tf_version = transformers_version()
    except Exception:
        tf_version = None

    # Don't take _gpu_lock here: warmup holds it for the whole weight load,
    # and the UI needs to poll this endpoint for the loading overlay.
    point = _point_segmenter
    video = _propagator

    return {
        "checkpoint": config.DEFAULT_CHECKPOINT,
        "architecture": "SAM 3 (Hugging Face transformers)",
        "point_model": "Sam3TrackerModel",
        "video_model": "Sam3TrackerVideoModel",
        "point_loaded": point is not None,
        "video_loaded": video is not None,
        "load_stage": _load_state["stage"],
        "load_error": _load_state["error"],
        "ready": point is not None,
        "point_device": getattr(point, "device", None),
        "video_device": getattr(video, "device", None),
        "video_precision": (
            str(getattr(video, "session_dtype", "")).replace("torch.", "")
            if video is not None
            else None
        ),
        "embedding_cache_size": getattr(point, "embedding_cache_size", None),
        "cached_slices": len(getattr(point, "_prepared", {})),
        "hardware": {
            "cuda_available": cuda_ok,
            "device_name": device_name or "CPU",
            "compute_capability": capability,
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu_count": torch.cuda.device_count() if cuda_ok else 0,
            "vram": vram,
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": tf_version,
        },
    }


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
    segmenter = _require_point_segmenter()
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
    segmenter = _require_point_segmenter()
    started = time.perf_counter()
    with _gpu_lock:
        mask = segmenter.segment(
            rgb,
            [(int(c), int(r)) for c, r in req.points],
            [int(l) for l in req.labels],
            image_key=(req.file, req.axis, req.index),
        )
    return {
        "mask": _mask_png_base64(mask, req.object_id),
        "coverage": float(mask.mean()),
        "timings": {
            **{k: float(v) for k, v in segmenter.last_timings.items()},
            "server_total": time.perf_counter() - started,
        },
    }


@app.post("/api/propagate")
def propagate(req: PropagateRequest) -> StreamingResponse:
    for obj in req.objects:
        if len(obj.points) != len(obj.labels):
            raise HTTPException(422, "points and labels must be equal length")
    data, geometry, data_u8 = _get_file(req.file)
    if geometry.kind != "3d":
        raise HTTPException(422, "Propagation requires a 3D volume")
    _validate_slice(data, geometry, req.axis, req.index)
    n_frames = _axis_count(data, geometry, req.axis)
    frames = [_slice_rgb(data_u8, geometry, req.axis, i) for i in range(n_frames)]
    propagator = _require_propagator()
    object_ids = [obj.id for obj in req.objects]

    events: queue.Queue[dict | None] = queue.Queue(maxsize=32)

    def on_progress(done: int, total: int, frame_idx: int, masks: np.ndarray) -> None:
        events.put(
            {
                "type": "frame",
                "frame": int(frame_idx),
                "done": int(done),
                "total": int(total),
                "coverage": float(masks.any(axis=0).mean()),
                "mask": _masks_png_base64(masks, object_ids),
            }
        )

    def run() -> None:
        try:
            with _gpu_lock:
                propagator.propagate(
                    frames,
                    anchor_idx=req.index,
                    points_per_object=[
                        [(int(c), int(r)) for c, r in obj.points]
                        for obj in req.objects
                    ],
                    labels_per_object=[
                        [int(l) for l in obj.labels] for obj in req.objects
                    ],
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
