"""Persistent SAM inference API for the React interactive client.

Run with:
    uvicorn server.main:app --host 127.0.0.1 --port 8000

The models stay resident between requests (mirroring the SAM 2 demo's
client/server split), so a point click only runs the prompt encoder and
mask decoder. Propagation streams one NDJSON event per tracked slice so
the browser can display and scrub frames while the sweep is still
running.

SAM 3 is loaded by default. POST /api/model with ``{"family": "sam2"}``,
``{"family": "sam3"}``, or ``{"family": "sam31"}`` to switch; only one
family stays on the GPU.
"""

from __future__ import annotations

import base64
import dataclasses
import gc
import io
import json
import logging
import math
import platform
import queue
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from seismic_app import config, sysinfo
from seismic_app.config import ModelFamily
from seismic_app.geometry import SectionGeometry
from seismic_app.inference import (
    LiveTracker,
    Sam31PointSegmenter,
    Sam31VolumePropagator,
    Sam3PointSegmenter,
    Sam3SeismicSegmenter,
    Sam3VolumePropagator,
    build_sam31_predictor,
    transformers_version,
)
from seismic_app.logutil import get_logger
from seismic_app.preprocessing import normalize_to_uint8
from seismic_app.pipeline import _segment_section_rgb
from seismic_app.sgy_loader import (
    SUPPORTED_SEISMIC_SUFFIXES,
    inspect_any_for_listing,
    load_any,
)
from seismic_app.stitching import binarize
from seismic_app.volume import (
    AxisNotReady,
    CachedVolume,
    DirectVolume,
    InMemoryVolume,
    SeismicVolume,
    estimated_in_memory_bytes,
    in_memory_budget,
    make_pages,
    needs_cache,
    slice_rgb,
)
from seismic_app.volume_cache import CacheManager
from seismic_app.vtk_export import FIRST_INTERACTIVE_LABEL_ID, export_volume_vti

log = get_logger("server")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

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


# One lock serializes GPU work; file loads use a separate cache lock so a
# tracker switch cannot freeze opening another .sgy / .npy survey.
_gpu_lock = threading.Lock()
_cache_lock = threading.Lock()
_family_lock = threading.Lock()
_FILE_CACHE_LIMIT = 2
_file_cache: OrderedDict[str, tuple[np.ndarray, SectionGeometry, np.ndarray]] = (
    OrderedDict()
)
_file_load_locks_guard = threading.Lock()
_file_load_locks: dict[str, threading.Lock] = {}
_point_segmenter: Sam3PointSegmenter | Sam31PointSegmenter | None = None
_propagator: Sam3VolumePropagator | Sam31VolumePropagator | None = None
_text_segmenter: Sam3SeismicSegmenter | None = None
_sam31_predictor = None
_active_family: ModelFamily = config.DEFAULT_MODEL_FAMILY
_load_id = 0
_load_state: dict[str, str | bool | None] = {
    "stage": "starting",
    "error": None,
}


class Propagation:
    """The last completed propagation: its live tracker plus what it covers.

    Held so that clicks on any slice can be answered as refinements of
    the tracked volume, and so the volume can be exported to ParaView
    without re-running anything.
    """

    def __init__(
        self,
        file: str,
        axis: str,
        object_ids: list[int],
        live: LiveTracker,
        start: int = 0,
        stop: int | None = None,
    ):
        self.file = file
        self.axis = axis
        self.object_ids = object_ids
        self.live = live
        # Tracked slice range [start, stop) along ``axis``; the live
        # tracker's frame i is slice start + i.
        self.start = start
        self.stop = stop if stop is not None else start + int(live.n_frames)

    def covers(self, file: str, axis: str) -> bool:
        return self.file == file and self.axis == axis

    def local_frame(self, index: int) -> int | None:
        """Live-tracker frame of an absolute slice index, None when outside the range."""
        if self.start <= index < self.stop:
            return index - self.start
        return None

    def position_of(self, object_id: int) -> int | None:
        """Row of a frontend object id in the tracked mask stack."""
        try:
            return self.object_ids.index(object_id)
        except ValueError:
            return None


_propagation: Propagation | None = None


def _require_propagation(file: str, axis: str) -> Propagation:
    if _propagation is None or not _propagation.covers(file, axis):
        raise HTTPException(
            409,
            "No tracked volume for this file and axis yet - propagate first.",
        )
    return _propagation


def _family_label(family: str | None = None) -> str:
    return str(config.family_spec(family or _active_family)["label"])


def _unload_models_locked() -> None:
    """Drop resident trackers so a different family can occupy the GPU."""
    global _point_segmenter, _propagator, _propagation, _sam31_predictor
    _propagation = None
    if _propagator is not None:
        close = getattr(_propagator, "close", None)
        if close is not None:
            close()
        else:
            _propagator.live = None
        _propagator.model = None
        _propagator.processor = None
        _propagator = None
    if _point_segmenter is not None:
        close = getattr(_point_segmenter, "close", None)
        if close is not None:
            close()
        else:
            _point_segmenter._prepared.clear()
        _point_segmenter.model = None
        _point_segmenter.processor = None
        _point_segmenter = None
    if _sam31_predictor is not None:
        _sam31_predictor.shutdown()
        _sam31_predictor.model = None
        _sam31_predictor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_models(family: ModelFamily, load_id: int) -> None:
    """Load one tracker family; ignore the result if a newer switch superseded it."""
    global _point_segmenter, _propagator, _sam31_predictor
    label = _family_label(family)
    try:
        with _gpu_lock:
            if load_id != _load_id:
                return
            _unload_models_locked()
            _load_state["error"] = None
            if family == "sam31":
                # The SAM 3 text detector is hidden while SAM 3.1 is active;
                # release it so the larger native multiplex model has VRAM.
                _unload_text_segmenter_locked()
                _load_state["stage"] = (
                    "Loading SAM 3.1 Object Multiplex onto the GPU..."
                )
                _sam31_predictor = build_sam31_predictor()
                _point_segmenter = Sam31PointSegmenter(
                    _sam31_predictor, embedding_cache_size=1
                )
                _propagator = Sam31VolumePropagator(_sam31_predictor)
            else:
                _load_state["stage"] = f"Loading {label} point tracker onto the GPU..."
                _point_segmenter = Sam3PointSegmenter(
                    family=family, embedding_cache_size=4
                )
            if load_id != _load_id:
                _unload_models_locked()
                return
            if family != "sam31":
                _load_state["stage"] = f"Loading {label} volume tracker onto the GPU..."
                _propagator = Sam3VolumePropagator(family=family)
            if load_id != _load_id:
                _unload_models_locked()
                return
            _load_state["stage"] = "ready"
            log.info(
                "Model warmup complete; %s point and volume trackers are resident.",
                label,
            )
    except Exception as exc:
        log.exception("Model warmup failed for %s", label)
        _load_state["error"] = str(exc)
        _load_state["stage"] = "ready" if _point_segmenter is not None else "error"


def _warmup_models() -> None:
    """Load the default tracker family at process start so the first click is warm."""
    global _load_id
    with _family_lock:
        # Don't stomp a family the UI already requested during startup.
        if _load_id != 0:
            return
        _load_id = 1
        load_id = _load_id
        family = _active_family
    _load_models(family, load_id)


def _request_family(family: ModelFamily) -> dict:
    """Switch the resident tracker family, loading in the background if needed."""
    global _active_family, _load_id
    spec = config.family_spec(family)
    with _family_lock:
        already = (
            family == _active_family
            and _point_segmenter is not None
            and _load_state["stage"] == "ready"
        )
        if already:
            return {"family": family, "ready": True, "label": spec["label"]}
        same_in_flight = family == _active_family and _load_state["stage"] not in (
            "ready",
            "error",
        )
        if same_in_flight:
            return {"family": family, "ready": False, "label": spec["label"]}
        _active_family = family
        _load_id += 1
        load_id = _load_id
        _load_state["error"] = None
        _load_state["stage"] = f"Switching to {spec['label']}..."
    threading.Thread(
        target=_load_models,
        args=(family, load_id),
        daemon=True,
        name=f"model-load-{family}",
    ).start()
    return {"family": family, "ready": False, "label": spec["label"]}


def _unload_text_segmenter_locked() -> None:
    global _text_segmenter
    if _text_segmenter is not None:
        _text_segmenter.model = None
        _text_segmenter.processor = None
        _text_segmenter = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_text_segmenter_locked() -> Sam3SeismicSegmenter:
    global _text_segmenter
    checkpoint = config.text_checkpoint()
    if _text_segmenter is not None and getattr(_text_segmenter, "checkpoint", None) == checkpoint:
        return _text_segmenter
    if _text_segmenter is not None:
        _unload_text_segmenter_locked()
    log.info("Loading text-prompt detector from %s", checkpoint)
    _text_segmenter = Sam3SeismicSegmenter(
        checkpoint=checkpoint,
        prompts=[config.FACIES_PROMPT],
    )
    return _text_segmenter


def _ensure_text_segmenter_locked() -> tuple[Sam3SeismicSegmenter, bool]:
    """Load Sam3Model for facies detection. Caller must hold ``_gpu_lock``.

    Returns (segmenter, unloaded_tracker). On CUDA OOM the click/volume
    trackers are dropped so the detector can occupy the GPU.
    """
    if _text_segmenter is not None:
        return _text_segmenter, False
    try:
        return _load_text_segmenter_locked(), False
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        log.warning(
            "CUDA OOM loading text detector alongside the tracker; "
            "unloading tracker so facies detection can run."
        )
        _unload_models_locked()
        _load_state["stage"] = "ready"
        _load_state["error"] = None
        return _load_text_segmenter_locked(), True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warmup_models, daemon=True, name="model-warmup").start()
    # CPU/disk only: never takes _gpu_lock, so model warmup is unaffected.
    threading.Thread(
        target=_prepare_large_files, daemon=True, name="large-file-scan"
    ).start()
    yield
    _cache_manager.shutdown()


app = FastAPI(title="Seismic SAM interactive API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    """Browser-friendly status for the API process (not the React UI)."""
    return {
        "service": "Seismic SAM interactive API",
        "frontend": "http://127.0.0.1:5173",
        "docs": "/docs",
        "runtime": "/api/runtime",
        "load_stage": _load_state["stage"],
        "load_error": _load_state["error"],
        "ready": _point_segmenter is not None and _load_state["stage"] != "error",
    }


def _list_seismic_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        (
            p
            for p in DATA_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SEISMIC_SUFFIXES
        ),
        key=lambda p: p.name.lower(),
    )


def _lock_for_file(name: str) -> threading.Lock:
    with _file_load_locks_guard:
        return _file_load_locks.setdefault(name, threading.Lock())


def _drop_tracked_volume_if_other_file_locked(file: str) -> None:
    """Close a live tracker session that belongs to a different survey."""
    global _propagation
    if _propagation is None or _propagation.file == file:
        return
    live = _propagation.live
    _propagation = None
    if _propagator is not None:
        _propagator.live = None
    closer = getattr(live, "close", None)
    if closer is not None:
        closer()


def _get_file(name: str) -> tuple[np.ndarray, SectionGeometry, np.ndarray]:
    allowed = {p.name: p for p in _list_seismic_files()}
    if name not in allowed:
        raise HTTPException(404, f"Unknown seismic file: {name}")
    with _cache_lock:
        cached = _file_cache.get(name)
        if cached is not None:
            _file_cache.move_to_end(name)
            return cached
    # Load outside the cache lock so one huge volume cannot block another file.
    with _lock_for_file(name):
        with _cache_lock:
            cached = _file_cache.get(name)
            if cached is not None:
                _file_cache.move_to_end(name)
                return cached
        log.info("Loading and normalizing %s...", name)
        try:
            data, geometry = load_any(allowed[name])
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(
                422, f"Could not load seismic file {name}: {exc}"
            ) from exc
        packed = (data, geometry, normalize_to_uint8(data))
        with _cache_lock:
            _file_cache[name] = packed
            _file_cache.move_to_end(name)
            while len(_file_cache) > _FILE_CACHE_LIMIT:
                evicted_name, _evicted = _file_cache.popitem(last=False)
                log.info("Evicted %s from the seismic file cache", evicted_name)
            return _file_cache.get(name, packed)


# ---- large files: routing, disk cache, volumes ------------------------------

_cache_manager = CacheManager()
_route_lock = threading.Lock()
_routes: dict[str, tuple[tuple, bool, tuple]] = {}
_volumes_lock = threading.Lock()
_large_volumes: dict[str, SeismicVolume] = {}


def _route(path: Path) -> tuple[bool, tuple[tuple[int, ...], SectionGeometry]]:
    """(needs the disk cache, fast listing) for one file version; logged once."""
    stat = path.stat()
    key = (stat.st_size, stat.st_mtime_ns)
    with _route_lock:
        cached = _routes.get(path.name)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
    listing = inspect_any_for_listing(path)
    shape = listing[0]
    # math.prod: np.prod overflows int32 on Windows for multi-billion-sample volumes.
    n_values = math.prod(int(s) for s in shape)
    large = needs_cache(n_values)
    log.info(
        "Routing %s (%s on disk, %s %s = %d values): in-memory cost ~%s vs budget %s -> %s",
        path.name,
        sysinfo.gb(stat.st_size),
        listing[1].kind.upper(),
        "x".join(str(s) for s in shape),
        n_values,
        sysinfo.gb(estimated_in_memory_bytes(n_values)),
        sysinfo.gb(in_memory_budget()),
        "disk cache (large-file path)" if large else "in-memory",
    )
    with _route_lock:
        _routes[path.name] = (key, large, listing)
    return large, listing


def _large_volume(name: str, path: Path, priority: bool = False, retry: bool = False) -> SeismicVolume:
    """Cached volume when the cache is complete, otherwise a direct reader."""
    state = _cache_manager.ensure(name, path, priority=priority, retry=retry)
    with _volumes_lock:
        current = _large_volumes.get(name)
        if state.ready:
            if isinstance(current, CachedVolume) and current.state is state:
                return current
            volume: SeismicVolume = CachedVolume(name, path, state)
            log.info("%s now served from its memmap cache (%s)", name, state.directory)
        else:
            if isinstance(current, DirectVolume) and current.state is state:
                return current
            _, listing = _route(path)
            volume = DirectVolume(name, path, state, listing)
        if current is not None:
            closer = getattr(current, "close", None)
            if closer is not None:
                closer()
        _large_volumes[name] = volume
        return volume


def _get_volume(name: str, priority: bool = False, retry: bool = False) -> SeismicVolume:
    """The slice source for a survey: in-memory for small files, cached for large."""
    path = {p.name: p for p in _list_seismic_files()}.get(name)
    if path is not None:
        try:
            large, _listing = _route(path)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(422, f"Could not inspect seismic file {name}: {exc}") from exc
        if large:
            return _large_volume(name, path, priority=priority, retry=retry)
    data, geometry, data_u8 = _get_file(name)
    return InMemoryVolume(name, data, geometry, data_u8)


def _prepare_large_files() -> None:
    """Startup pass: route every data file and queue caches for the large ones."""
    started = time.perf_counter()
    queued = []
    for path in _list_seismic_files():
        try:
            large, _ = _route(path)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            log.warning("Could not inspect %s at startup: %s", path.name, exc)
            continue
        if large:
            state = _cache_manager.ensure(path.name, path)
            queued.append(f"{path.name} ({state.stage})")
    log.info(
        "Large-file startup scan done in %.1fs: %s",
        time.perf_counter() - started,
        ", ".join(queued) if queued else "no files need the disk cache",
    )


def _require_point_segmenter() -> Sam3PointSegmenter | Sam31PointSegmenter:
    if _load_state["error"] and _point_segmenter is None:
        raise HTTPException(503, f"Model failed to load: {_load_state['error']}")
    if (
        _point_segmenter is None
        or getattr(_point_segmenter, "family", None) != _active_family
    ):
        raise HTTPException(
            503,
            str(
                _load_state["stage"]
                or f"{_family_label()} point tracker is still loading"
            ),
        )
    return _point_segmenter


def _require_propagator() -> Sam3VolumePropagator | Sam31VolumePropagator:
    if _load_state["error"] and _propagator is None:
        raise HTTPException(503, f"Model failed to load: {_load_state['error']}")
    if (
        _propagator is None
        or getattr(_propagator, "family", None) != _active_family
    ):
        raise HTTPException(
            503,
            str(
                _load_state["stage"]
                or f"{_family_label()} volume tracker is still loading"
            ),
        )
    return _propagator


def _slice_rgb(
    data_u8: np.ndarray, geometry: SectionGeometry, axis: str, index: int
) -> np.ndarray:
    """Slice image of an in-memory uint8 array (index = page for wide 2D lines)."""
    return slice_rgb(data_u8, geometry, axis, index)


def _volume_rgb(volume: SeismicVolume, axis: str, index: int) -> np.ndarray:
    """(H, W, 3) uint8 slice image; 409 when that axis of a large file is not ready."""
    if isinstance(volume, InMemoryVolume):
        return _slice_rgb(volume.data_u8, volume.geometry, axis, index)
    try:
        return volume.rgb(axis, index)
    except AxisNotReady as exc:
        raise HTTPException(409, str(exc)) from exc


def _validate_slice(volume: SeismicVolume, axis: str, index: int) -> None:
    if axis not in ("inline", "crossline", "time"):
        raise HTTPException(422, f"Unknown axis: {axis}")
    count = volume.axis_count(axis)
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


def _facies_mask_png_base64(mask: np.ndarray) -> str:
    color = config.LABEL_COLORS[config.FACIES_PROMPT]
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, MASK_ALPHA)
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
    # Per-point slice index (SAM2-style refinement clicks on any slice).
    # Defaults to the request's anchor slice for every point.
    slices: list[int] | None = None


class PropagateRequest(SliceRef):
    objects: list[ObjectPrompt] = Field(min_length=1)
    # Half-width of the tracked slice range around the anchor. None tracks
    # the whole axis when it fits the memory budget (always for small files).
    window: int | None = Field(default=None, ge=0)


class RefineRequest(SliceRef):
    """Clicks for one object on one slice of an already-tracked volume."""

    object_id: int = 0
    points: list[list[int]] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)


class VolumeRef(BaseModel):
    file: str
    axis: str = "inline"


class ExportRequest(VolumeRef):
    include_amplitude: bool = True


class SetModelRequest(BaseModel):
    family: str


class SetTextCheckpointRequest(BaseModel):
    checkpoint: str


@app.post("/api/model")
def set_model(req: SetModelRequest) -> dict:
    """Switch the resident tracker between SAM 2, SAM 3, and SAM 3.1.

    Only one family is kept on the GPU. The previous models, cached
    embeddings, and any live tracked volume are dropped. Loading runs in
    the background; poll ``/api/runtime`` until ``ready`` is true.
    """
    try:
        family = config.resolve_family(req.family)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _request_family(family)


@app.post("/api/text-checkpoint")
def set_text_checkpoint(req: SetTextCheckpointRequest) -> dict:
    """Switch the facies text detector to a converted local folder or official SAM 3.

    Does not touch the click/volume tracker. The previous detector is
    dropped; the new weights load on the next Detect request.
    """
    try:
        checkpoint = config.set_text_checkpoint(req.checkpoint)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with _gpu_lock:
        loaded = getattr(_text_segmenter, "checkpoint", None)
        if loaded != checkpoint:
            _unload_text_segmenter_locked()
    label = Path(checkpoint).name if Path(checkpoint).exists() else checkpoint
    return {"checkpoint": checkpoint, "label": label, "loaded": False}


@app.get("/api/files")
def list_files(format: str | None = None) -> list[dict]:
    """List data metadata; an optional format filter supports fast NPY discovery."""
    if format is not None and format.lower() not in ("npy", "sgy"):
        raise HTTPException(422, "format must be 'npy' or 'sgy'")
    requested_suffix = f".{format.lower()}" if format else None
    entries = []
    for path in _list_seismic_files():
        if requested_suffix is not None and path.suffix.lower() != requested_suffix:
            continue
        try:
            large, (shape, geometry) = _route(path)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            log.warning("Skipping unusable data file %s: %s", path, exc)
            continue
        entry = {
            "name": path.name,
            "format": path.suffix.lower().lstrip("."),
            "kind": geometry.kind,
            "shape": list(shape),
            "axes": {
                "inline": shape[0],
                "crossline": shape[1],
                "time": shape[2],
            }
            if geometry.kind == "3d"
            else {"inline": 1, "crossline": 1, "time": 1},
        }
        # Large-file fields (large, status, axes_ready, pages) are only added
        # where they apply, so small-file entries keep their original shape.
        if geometry.kind == "2d":
            pages = make_pages(int(shape[1]))
            if pages.count > 1:
                entry["pages"] = pages.to_json()
        if large:
            volume = _large_volume(path.name, path)
            entry.update(volume.info())
        entries.append(entry)
    return entries


@app.get("/api/meta")
def file_meta(file: str) -> dict:
    """Return kind/shape for a survey. Does not require a loaded tracker.

    Large files answer immediately (their cache builds in the background,
    and opening one moves it to the front of the queue); small files are
    loaded into memory as before.
    """
    started = time.perf_counter()
    volume = _get_volume(file, priority=True, retry=True)
    info = volume.info()
    log.info(
        "Opened %s: %s %s%s [%.2fs]",
        file,
        info["kind"].upper(),
        "x".join(str(s) for s in info["shape"]),
        f", cache {info['status']['stage']} {info['status']['percent']:.0f}%"
        if info.get("status")
        else " (in memory)",
        time.perf_counter() - started,
    )
    return info


@app.get("/api/file-status")
def file_status(file: str) -> dict:
    """Cache build progress and axis readiness for one survey (cheap to poll)."""
    path = {p.name: p for p in _list_seismic_files()}.get(file)
    if path is None:
        raise HTTPException(404, f"Unknown seismic file: {file}")
    try:
        large, (shape, geometry) = _route(path)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, f"Could not inspect seismic file {file}: {exc}") from exc
    if not large:
        return {
            "name": file,
            "large": False,
            "kind": geometry.kind,
            "shape": list(shape),
            "axes_ready": {"inline": True, "crossline": True, "time": True},
            "status": None,
        }
    return _large_volume(file, path).info()


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

    spec = config.family_spec(_active_family)
    loaded_family = getattr(point, "family", None)
    models_match = loaded_family == _active_family
    # Clicks can start as soon as the point tracker of the requested family
    # is resident; the volume tracker may still be loading.
    point_ready = (
        point is not None and models_match and _load_state["stage"] != "error"
    )
    video_ready = (
        video is not None and models_match and _load_state["stage"] == "ready"
    )

    return {
        "family": _active_family,
        "family_label": spec["label"],
        "available_models": [
            {
                "id": item["id"],
                "label": item["label"],
                "checkpoint": item["checkpoint"],
                "gated": item["gated"],
            }
            for item in config.SAM_FAMILIES.values()
        ],
        "checkpoint": getattr(point, "checkpoint", None) or spec["checkpoint"],
        "text_checkpoint": config.text_checkpoint(),
        "available_text_checkpoints": config.list_text_checkpoints(),
        "text_detector_loaded": _text_segmenter is not None,
        "architecture": spec["architecture"],
        "point_model": spec["point_model"],
        "video_model": spec["video_model"],
        "point_loaded": point_ready,
        "video_loaded": video_ready,
        "load_stage": _load_state["stage"],
        "load_error": _load_state["error"],
        "ready": point_ready,
        "point_device": getattr(point, "device", None),
        "video_device": getattr(video, "device", None),
        "video_precision": (
            str(getattr(video, "session_dtype", "")).replace("torch.", "")
            if video is not None
            else None
        ),
        "embedding_cache_size": getattr(point, "embedding_cache_size", None),
        "cached_slices": len(getattr(point, "_prepared", {})),
        # Lets the UI restore the edit/export affordances after a reload.
        "tracked_volume": (
            {
                "file": _propagation.file,
                "axis": _propagation.axis,
                "objects": list(_propagation.object_ids),
                "edited_slices": sorted(
                    _propagation.start + f for f in _propagation.live.dirty_frames
                ),
                "start": _propagation.start,
                "stop": _propagation.stop,
            }
            if _propagation is not None
            else None
        ),
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
    started = time.perf_counter()
    volume = _get_volume(file)
    _validate_slice(volume, axis, index)
    rgb = _volume_rgb(volume, axis, index)
    read_done = time.perf_counter()
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=90)
    total = time.perf_counter() - started
    log.log(
        logging.INFO if total > 0.25 else logging.DEBUG,
        "slice %s %s[%d] %dx%d: read %.0f ms, jpeg %.0f ms (%s)",
        file,
        axis,
        index,
        rgb.shape[1],
        rgb.shape[0],
        1000 * (read_done - started),
        1000 * (total - (read_done - started)),
        type(volume).__name__,
    )
    return Response(
        buffer.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=3600"},
    )


@app.post("/api/prepare")
def prepare_slice(req: SliceRef) -> dict:
    """Encode the slice with the vision backbone before the first click."""
    volume = _get_volume(req.file)
    _validate_slice(volume, req.axis, req.index)
    rgb = _volume_rgb(volume, req.axis, req.index)
    with _gpu_lock:
        _drop_tracked_volume_if_other_file_locked(req.file)
        segmenter = _require_point_segmenter()
        segmenter.prepare_image(rgb, image_key=(req.file, req.axis, req.index))
        prepare_seconds = segmenter.last_timings.get("prepare_image", 0.0)
    return {
        "prepare_seconds": prepare_seconds,
        **volume.info(),
    }


@app.post("/api/segment")
def segment(req: SegmentRequest) -> dict:
    if len(req.points) != len(req.labels):
        raise HTTPException(422, "points and labels must be equal length")
    volume = _get_volume(req.file)
    _validate_slice(volume, req.axis, req.index)
    rgb = _volume_rgb(volume, req.axis, req.index)
    started = time.perf_counter()
    with _gpu_lock:
        segmenter = _require_point_segmenter()
        mask = segmenter.segment(
            rgb,
            [(int(c), int(r)) for c, r in req.points],
            [int(l) for l in req.labels],
            image_key=(req.file, req.axis, req.index),
            object_id=req.object_id,
        )
        timings = {k: float(v) for k, v in segmenter.last_timings.items()}
    return {
        "mask": _mask_png_base64(mask, req.object_id),
        "coverage": float(mask.mean()),
        "timings": {
            **timings,
            "server_total": time.perf_counter() - started,
        },
    }


@app.post("/api/auto-segment")
def auto_segment(req: SliceRef) -> dict:
    """Run the fine-tuned text detector for ``seismic facies`` on one slice.

    Uses ``Sam3SeismicSegmenter`` (not the click tracker). The tracker
    stays loaded when VRAM allows; otherwise it is unloaded for this pass.
    """
    volume = _get_volume(req.file)
    _validate_slice(volume, req.axis, req.index)
    rgb = _volume_rgb(volume, req.axis, req.index)
    started = time.perf_counter()
    prompts = [config.FACIES_PROMPT]
    with _gpu_lock:
        segmenter, unloaded_tracker = _ensure_text_segmenter_locked()
        try:
            prob_maps = _segment_section_rgb(rgb, segmenter, prompts)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            log.warning("CUDA OOM during facies detection; unloading tracker and retrying.")
            _unload_models_locked()
            unloaded_tracker = True
            _load_state["stage"] = "ready"
            segmenter = _load_text_segmenter_locked()
            prob_maps = _segment_section_rgb(rgb, segmenter, prompts)
    masks = binarize(prob_maps, config.MASK_THRESHOLD)
    mask = np.asarray(masks[config.FACIES_PROMPT], dtype=bool)
    return {
        "mask": _facies_mask_png_base64(mask),
        "coverage": float(mask.mean()),
        "prompt": config.FACIES_PROMPT,
        "checkpoint": config.text_checkpoint(),
        "unloaded_tracker": unloaded_tracker,
        "timings": {"server_total": time.perf_counter() - started},
    }


def _stream_tracking(
    object_ids: list[int],
    work: Callable[[Callable[[int, int, int, np.ndarray], None]], dict],
    frame_offset: int = 0,
) -> StreamingResponse:
    """Run a tracking job on a worker thread, one NDJSON event per slice.

    ``work`` is called with a progress callback and returns the payload
    to merge into the terminating "done" event. ``frame_offset`` turns the
    tracker's window-local frame numbers back into absolute slice indices.
    """
    events: queue.Queue[dict | None] = queue.Queue(maxsize=32)

    def on_progress(done: int, total: int, frame_idx: int, masks: np.ndarray) -> None:
        events.put(
            {
                "type": "frame",
                "frame": int(frame_idx) + frame_offset,
                "done": int(done),
                "total": int(total),
                "coverage": float(masks.any(axis=0).mean()),
                "mask": _masks_png_base64(masks, object_ids),
            }
        )

    def run() -> None:
        try:
            events.put({"type": "done", **work(on_progress)})
        except Exception as exc:  # surface tracker errors to the browser
            log.exception("Tracking failed")
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


def _propagation_window(
    volume: SeismicVolume,
    axis: str,
    anchor: int,
    prompted: list[int],
    n_objects: int,
    requested: int | None,
) -> tuple[int, int, str]:
    """Slice range [start, stop) to track, always covering every prompted slice.

    Small (in-memory) volumes track the whole axis unless a half-width is
    requested, exactly as before. Large volumes size the window from the
    RAM a propagation needs per frame: the RGB slice, its mask per object
    and the processor's resized float tensor.
    """
    n = volume.axis_count(axis)
    h, w = volume.frame_shape(axis)
    side = config.PROPAGATION_MODEL_SIDE
    per_frame = h * w * (3 + n_objects) + 3 * side * side * 4
    total, available = sysinfo.memory_status()
    budget = int(config.PROPAGATION_RAM_FRACTION * total) if total else 4 * 1024**3
    if available:
        budget = min(budget, int(available * 0.6))
    max_frames = max(config.PROPAGATION_MIN_FRAMES, budget // per_frame)
    if requested is not None and requested >= 0:
        start, stop = anchor - requested, anchor + requested + 1
        reason = f"requested +/-{requested}"
    elif not volume.large or n <= max_frames:
        start, stop = 0, n
        reason = "full axis" + ("" if volume.large else " (in-memory volume)")
    else:
        half = max(1, (max_frames - 1) // 2)
        start, stop = anchor - half, anchor + half + 1
        reason = f"auto +/-{half} from a {sysinfo.gb(budget)} RAM budget"
    touched = prompted + [anchor]
    start = max(0, min(start, min(touched)))
    stop = min(n, max(stop, max(touched) + 1))
    estimate = (stop - start) * per_frame
    log.info(
        "Propagation window on %s %s: slices [%d, %d) of %d (%s); ~%s for %d frames of %dx%d "
        "(budget %s, %s available)",
        volume.name,
        axis,
        start,
        stop,
        n,
        reason,
        sysinfo.gb(estimate),
        stop - start,
        w,
        h,
        sysinfo.gb(budget),
        sysinfo.gb(available),
    )
    if estimate > budget and volume.large:
        log.warning(
            "Requested propagation range needs ~%s, above the %s budget; it may be slow "
            "or run out of memory",
            sysinfo.gb(estimate),
            sysinfo.gb(budget),
        )
    return start, stop, reason


@app.post("/api/propagate")
def propagate(req: PropagateRequest) -> StreamingResponse:
    volume = _get_volume(req.file)
    if volume.kind != "3d":
        raise HTTPException(422, "Propagation requires a 3D volume")
    _validate_slice(volume, req.axis, req.index)
    if not volume.axes_ready().get(req.axis, False):
        raise HTTPException(
            409,
            f"The {req.axis} axis of {req.file} is not ready yet; wait for its cache to finish.",
        )
    n_frames = volume.axis_count(req.axis)
    for obj in req.objects:
        if len(obj.points) != len(obj.labels):
            raise HTTPException(422, "points and labels must be equal length")
        if 1 not in obj.labels:
            raise HTTPException(
                422,
                f"object {obj.id + 1} has only negative points; "
                "add at least one + point to define the object",
            )
        if obj.slices is not None:
            if len(obj.slices) != len(obj.points):
                raise HTTPException(422, "slices must match points length")
            if any(not 0 <= s < n_frames for s in obj.slices):
                raise HTTPException(422, "point slice index out of range")
    prompted = sorted(
        {int(s) for obj in req.objects for s in (obj.slices or [req.index])}
    )
    start, stop, reason = _propagation_window(
        volume, req.axis, req.index, prompted, len(req.objects), req.window
    )
    frames_started = time.perf_counter()
    frames = [_volume_rgb(volume, req.axis, i) for i in range(start, stop)]
    log.info(
        "Built %d %s frames for propagation in %.1fs",
        len(frames),
        req.axis,
        time.perf_counter() - frames_started,
    )
    object_ids = [obj.id for obj in req.objects]

    def work(on_progress) -> dict:
        global _propagation
        with _gpu_lock:
            propagator = _require_propagator()
            # Drop the previous volume while model switching is excluded, so
            # a completed stale job cannot restore an already-closed session.
            _propagation = None
            propagator.propagate(
                frames,
                anchor_idx=req.index - start,
                points_per_object=[
                    [(int(c), int(r)) for c, r in obj.points] for obj in req.objects
                ],
                labels_per_object=[
                    [int(l) for l in obj.labels] for obj in req.objects
                ],
                frame_indices_per_object=[
                    [int(s) - start for s in obj.slices]
                    if obj.slices is not None
                    else [req.index - start] * len(obj.points)
                    for obj in req.objects
                ],
                progress=on_progress,
                keep_live=True,
            )
            live = propagator.live
            if live is not None:
                _propagation = Propagation(
                    req.file, req.axis, object_ids, live, start=start, stop=stop
                )
            timings = {
                k: float(v) for k, v in propagator.last_timings.items()
            }
        return {
            "timings": timings,
            "editable": live is not None,
            "start": start,
            "stop": stop,
            "window_reason": reason,
        }

    return _stream_tracking(object_ids, work, frame_offset=start)


@app.post("/api/refine")
def refine(req: RefineRequest) -> dict:
    """Re-decode one slice for one object using the live tracker's memory.

    This is the interactive edit path: because the slice has already been
    tracked, a negative click carves into the propagated mask and a
    positive click extends it, both visible on this slice immediately.
    """
    if len(req.points) != len(req.labels):
        raise HTTPException(422, "points and labels must be equal length")
    volume = _get_volume(req.file)
    _validate_slice(volume, req.axis, req.index)
    started = time.perf_counter()
    with _gpu_lock:
        state = _require_propagation(req.file, req.axis)
        position = state.position_of(req.object_id)
        if position is None:
            raise HTTPException(
                409,
                f"Object {req.object_id + 1} is not part of the tracked volume - "
                "propagate again to include it.",
            )
        local = state.local_frame(req.index)
        if local is None:
            raise HTTPException(
                409,
                f"Slice {req.index + 1} is outside the tracked range "
                f"{state.start + 1}-{state.stop} - propagate with a wider range to edit it.",
            )
        propagator = _require_propagator()
        frame_masks = propagator.refine_frame(
            state.live,
            position,
            local,
            [(int(c), int(r)) for c, r in req.points],
            [int(l) for l in req.labels],
        )
    return {
        "mask": _masks_png_base64(frame_masks, state.object_ids),
        "coverage": float(frame_masks.any(axis=0).mean()),
        "timings": {"server_total": time.perf_counter() - started},
    }


@app.post("/api/resweep")
def resweep(req: VolumeRef) -> StreamingResponse:
    """Re-track the volume outward from the slices edited since the last sweep."""
    volume = _get_volume(req.file)
    if volume.kind != "3d":
        raise HTTPException(422, "Propagation requires a 3D volume")
    with _gpu_lock:
        state = _require_propagation(req.file, req.axis)
        if not state.live.dirty_frames:
            raise HTTPException(
                409, "Nothing was edited since the last propagation."
            )
        object_ids = list(state.object_ids)
        offset = state.start
        start, stop = state.start, state.stop

    def work(on_progress) -> dict:
        with _gpu_lock:
            current_state = _require_propagation(req.file, req.axis)
            if not current_state.live.dirty_frames:
                raise HTTPException(
                    409, "Nothing was edited since the last propagation."
                )
            current_propagator = _require_propagator()
            current_propagator.resweep(
                current_state.live, progress=on_progress
            )
            timings = {
                k: float(v) for k, v in current_propagator.last_timings.items()
            }
        return {
            "timings": timings,
            "editable": True,
            "start": start,
            "stop": stop,
        }

    return _stream_tracking(object_ids, work, frame_offset=offset)


def _slice_masks_to_cube(
    masks: np.ndarray, axis: str, cube_shape: tuple[int, int, int]
) -> np.ndarray:
    """(n_frames, H, W) per-slice masks -> an (n_il, n_xl, n_samples) volume.

    Undoes the per-axis orientation that _slice_rgb applies, so the
    exported labels land on the same samples the user clicked on.
    """
    n_il, n_xl, n_samples = cube_shape
    cube = np.zeros(cube_shape, dtype=bool)
    if axis == "inline":  # frame i = inline i, imaged (n_samples, n_xl)
        for i in range(min(masks.shape[0], n_il)):
            cube[i] = masks[i].T
    elif axis == "crossline":  # frame j = crossline j, imaged (n_samples, n_il)
        for j in range(min(masks.shape[0], n_xl)):
            cube[:, j, :] = masks[j].T
    else:  # frame k = time sample k, imaged (n_il, n_xl)
        for k in range(min(masks.shape[0], n_samples)):
            cube[:, :, k] = masks[k]
    return cube


@app.post("/api/export")
def export_volume(req: ExportRequest) -> dict:
    """Write a tracked SEG-Y or NumPy volume to a ParaView .vti.

    Only the tracked slice range is written, placed at its offset in the
    survey, so windowed propagations on large volumes export quickly.
    """
    volume = _get_volume(req.file)
    if volume.kind != "3d":
        raise HTTPException(422, "Volume export requires a 3D volume")
    state = _require_propagation(req.file, req.axis)
    masks = state.live.masks
    start, stop = state.start, state.stop

    started = time.perf_counter()
    sub_shape = list(volume.shape)
    sub_shape[{"inline": 0, "crossline": 1, "time": 2}[req.axis]] = stop - start
    sub_shape_t = (int(sub_shape[0]), int(sub_shape[1]), int(sub_shape[2]))
    # One named mask per object; vtk_export folds them into a single int
    # label array, assigning ids in insertion order. Later objects win
    # where two masks overlap.
    label_masks = {
        f"object {object_id + 1}": np.transpose(
            _slice_masks_to_cube(masks[row], req.axis, sub_shape_t), (0, 2, 1)
        )
        for row, object_id in enumerate(state.object_ids)
    }
    if req.include_amplitude:
        try:
            amplitude = volume.amplitude_window(req.axis, start, stop)
        except AxisNotReady as exc:
            raise HTTPException(409, str(exc)) from exc
    else:
        amplitude = np.zeros((0, 0, 0), dtype=np.float32)
    geometry, origin_xy = _window_geometry(volume.geometry, req.axis, start, stop)
    log.info(
        "Exporting %s %s slices [%d, %d) as a %s sub-volume%s",
        req.file,
        req.axis,
        start,
        stop,
        "x".join(str(s) for s in sub_shape_t),
        " with amplitude" if req.include_amplitude else "",
    )
    out_base = _export_path(req.file, req.axis).with_suffix("")
    written = export_volume_vti(
        label_masks,
        amplitude,
        geometry,
        out_base,
        include_amplitude=req.include_amplitude,
        origin_xy=origin_xy,
    )
    elapsed = time.perf_counter() - started
    legend = {
        f"object {object_id + 1}": FIRST_INTERACTIVE_LABEL_ID + row
        for row, object_id in enumerate(state.object_ids)
    }
    log.info("Exported %s in %.1fs", written, elapsed)
    return {
        "path": str(written),
        "directory": str(written.parent),
        "size_mb": round(written.stat().st_size / (1024**2), 1),
        "labels": legend,
        "seconds": elapsed,
    }


def _window_geometry(
    geometry: SectionGeometry, axis: str, start: int, stop: int
) -> tuple[SectionGeometry, tuple[float, float]]:
    """Geometry of a [start, stop) sub-volume and its (x, y) origin in metres."""
    il_sp = geometry.iline_spacing_m or 25.0
    xl_sp = geometry.xline_spacing_m or 25.0
    if axis == "inline":
        ilines = geometry.ilines[start:stop] if geometry.ilines is not None else None
        return dataclasses.replace(geometry, ilines=ilines), (start * il_sp, 0.0)
    if axis == "crossline":
        xlines = geometry.xlines[start:stop] if geometry.xlines is not None else None
        return dataclasses.replace(geometry, xlines=xlines), (0.0, start * xl_sp)
    return (
        dataclasses.replace(
            geometry,
            n_samples=stop - start,
            t0_ms=geometry.t0_ms + start * geometry.dt_ms,
        ),
        (0.0, 0.0),
    )


def _export_path(file: str, axis: str) -> Path:
    return OUTPUT_DIR / f"{Path(file).stem}_{axis}_objects.vti"


@app.get("/api/export/file")
def download_export(file: str, axis: str = "inline") -> FileResponse:
    """Stream the last written .vti so the browser can save it for ParaView."""
    allowed = {p.name for p in _list_seismic_files()}
    if file not in allowed:
        raise HTTPException(404, f"Unknown seismic file: {file}")
    if axis not in ("inline", "crossline", "time"):
        raise HTTPException(422, f"Unknown axis: {axis}")
    written = _export_path(file, axis)
    if not written.is_file():
        raise HTTPException(
            404, "No export file yet - export the tracked volume first."
        )
    return FileResponse(
        written,
        media_type="application/octet-stream",
        filename=written.name,
    )
