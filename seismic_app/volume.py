"""One slice-serving interface for small in-memory files and large cached ones.

The API and the SAM trackers only ever need (H, W, 3) uint8 slice
images plus, for export, float amplitudes of a sub-volume. ``SeismicVolume``
provides exactly that, whatever the storage:

- ``InMemoryVolume`` - today's path for files that fit in RAM; slices are
  produced by the same functions as before, so they are byte-identical.
- ``DirectVolume`` - a large file whose cache is still being built.
  Slices along the storage axis (inline for inline-sorted files) are
  read straight from the source and normalized with the sampled clip
  bounds; the other axes unlock when the cache is ready.
- ``CachedVolume`` - a large file served from the uint8 memmaps; every
  slice is a contiguous read, RAM use stays near zero.

Large 2D lines are served as overlapping pages of traces so that no
slice image is wider than ``config.MAX_2D_PAGE_TRACES``.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import config, sysinfo
from .geometry import SectionGeometry
from .logutil import get_logger
from .preprocessing import inline_to_rgb_25d, to_rgb
from .segy_geometry import GridLayout, cells_for_traces
from .sources import SeismicSource, open_source
from .volume_cache import BuildState, load_geometry, normalize_block, open_memmaps

log = get_logger("volume")

AXES = ("inline", "crossline", "time")


class AxisNotReady(Exception):
    """A slice was requested along an axis the cache has not built yet."""

    def __init__(self, message: str, status: dict | None = None):
        super().__init__(message)
        self.status = status or {}


# ---- routing -----------------------------------------------------------------


def in_memory_budget() -> int:
    if config.IN_MEMORY_BUDGET_GB:
        return int(float(config.IN_MEMORY_BUDGET_GB) * 1024**3)
    total = sysinfo.total_ram() or 16 * 1024**3
    return int(total * config.IN_MEMORY_BUDGET_FRACTION)


def estimated_in_memory_bytes(n_values: int) -> int:
    return int(n_values) * config.IN_MEMORY_BYTES_PER_VALUE


def needs_cache(n_values: int) -> bool:
    return estimated_in_memory_bytes(n_values) > in_memory_budget()


# ---- 2D pages ----------------------------------------------------------------


@dataclass(frozen=True)
class Pages:
    """Overlapping windows of traces for 2D lines too wide to display at once."""

    count: int
    width: int
    step: int
    n_traces: int

    def bounds(self, page: int) -> tuple[int, int]:
        start = min(page * self.step, self.n_traces - self.width)
        return start, start + self.width

    def to_json(self) -> dict:
        return {"count": self.count, "width": self.width, "step": self.step}


def make_pages(n_traces: int) -> Pages:
    limit = config.MAX_2D_PAGE_TRACES
    if n_traces <= limit:
        return Pages(1, n_traces, n_traces, n_traces)
    step = max(1, limit - config.PAGE_OVERLAP_TRACES)
    count = math.ceil((n_traces - limit) / step) + 1
    return Pages(count, limit, step, n_traces)


# ---- display orientation (shared by every implementation) -------------------


def slice_rgb(data_u8: np.ndarray, geometry: SectionGeometry, axis: str, index: int) -> np.ndarray:
    """The app's slice image convention for an in-memory uint8 array."""
    if geometry.kind == "2d":
        pages = make_pages(data_u8.shape[1])
        if pages.count == 1:
            return to_rgb(data_u8)
        a, b = pages.bounds(index)
        return to_rgb(data_u8[:, a:b])
    if axis == "inline":
        return inline_to_rgb_25d(data_u8, index)
    if axis == "crossline":
        return to_rgb(data_u8[:, index, :].T)
    return to_rgb(data_u8[:, :, index])


def _rgb_from_planes(axis: str, index: int, count: int, plane) -> np.ndarray:
    """Build the slice image from ``plane(axis, i)`` (cube-order 2D planes)."""
    if axis == "inline":
        prev_idx = max(0, index - 1)
        next_idx = min(count - 1, index + 1)
        return np.stack(
            [plane(axis, prev_idx).T, plane(axis, index).T, plane(axis, next_idx).T],
            axis=-1,
        )
    if axis == "crossline":
        return to_rgb(plane(axis, index).T)
    return to_rgb(plane(axis, index))


# ---- the interface -----------------------------------------------------------


class SeismicVolume:
    name: str
    format: str
    kind: str
    geometry: SectionGeometry
    shape: tuple[int, ...]
    large: bool = False

    @property
    def pages(self) -> Pages | None:
        if self.kind != "2d":
            return None
        return make_pages(self.shape[1])

    def axis_count(self, axis: str) -> int:
        if self.kind == "2d":
            return self.pages.count if self.pages else 1
        return {"inline": self.shape[0], "crossline": self.shape[1], "time": self.shape[2]}[axis]

    def frame_shape(self, axis: str) -> tuple[int, int]:
        """(H, W) of slice images along ``axis``."""
        if self.kind == "2d":
            pages = self.pages
            return (self.shape[0], pages.width if pages else self.shape[1])
        n_il, n_xl, n_s = self.shape
        return {"inline": (n_s, n_xl), "crossline": (n_s, n_il), "time": (n_il, n_xl)}[axis]

    def axes_ready(self) -> dict[str, bool]:
        return {axis: True for axis in AXES}

    def rgb(self, axis: str, index: int) -> np.ndarray:
        raise NotImplementedError

    def amplitude_window(self, axis: str, start: int, stop: int) -> np.ndarray:
        """Float amplitudes (n_il, n_xl, n_s) restricted to [start, stop) along ``axis``."""
        raise NotImplementedError

    def status(self) -> dict | None:
        return None

    def info(self) -> dict:
        """API metadata. Small files keep the original keys; extras are added
        only when they apply (large files, paged 2D lines)."""
        shape = [int(size) for size in self.shape]
        pages = self.pages
        info = {
            "name": self.name,
            "format": self.format,
            "kind": self.kind,
            "shape": shape,
            "axes": {
                "inline": shape[0],
                "crossline": shape[1],
                "time": shape[2],
            }
            if self.kind == "3d"
            else {"inline": 1, "crossline": 1, "time": 1},
        }
        if pages and pages.count > 1:
            info["pages"] = pages.to_json()
        if self.large:
            info["large"] = True
            info["axes_ready"] = self.axes_ready()
            info["status"] = self.status()
        return info


class InMemoryVolume(SeismicVolume):
    def __init__(self, name: str, data: np.ndarray, geometry: SectionGeometry, data_u8: np.ndarray):
        self.name = name
        self.format = Path(name).suffix.lower().lstrip(".")
        self.kind = geometry.kind
        self.geometry = geometry
        self.data = data
        self.data_u8 = data_u8
        self.shape = tuple(int(s) for s in data.shape)

    def rgb(self, axis: str, index: int) -> np.ndarray:
        return slice_rgb(self.data_u8, self.geometry, axis, index)

    def amplitude_window(self, axis: str, start: int, stop: int) -> np.ndarray:
        if axis == "inline":
            return self.data[start:stop]
        if axis == "crossline":
            return self.data[:, start:stop]
        return self.data[:, :, start:stop]


# ---- sub-volume amplitude reads for export ------------------------------------


def read_subcube(
    source: SeismicSource,
    layout: GridLayout,
    cellmap: np.ndarray | None,
    axis: str,
    start: int,
    stop: int,
) -> np.ndarray:
    """Float32 (n_il, n_xl, n_s) window of a large 3D source along one axis."""
    n_il, n_xl, n_s = layout.shape
    width = stop - start
    shape = {
        "inline": (width, n_xl, n_s),
        "crossline": (n_il, width, n_s),
        "time": (n_il, n_xl, width),
    }[axis]
    started = time.perf_counter()
    out = np.zeros(shape, dtype=np.float32)
    if layout.regular and layout.sort == "inline" and axis == "inline":
        out[:] = source.read(start * n_xl, stop * n_xl).reshape(width, n_xl, n_s)
    elif layout.regular and layout.sort == "crossline" and axis == "crossline":
        block = source.read(start * n_il, stop * n_il).reshape(width, n_il, n_s)
        out[:] = block.transpose(1, 0, 2)
    else:
        n = layout.n_traces
        block_traces = max(1, config.CACHE_BLOCK_BYTES // (n_s * 4))
        if axis == "time":
            log.info(
                "Reading all %d traces for a %d-sample time window export", n, width
            )
        for a in range(0, n, block_traces):
            b = min(n, a + block_traces)
            cells = cells_for_traces(layout, cellmap, a, b)
            il_idx, xl_idx = np.divmod(cells, n_xl)
            valid = cells >= 0
            if axis == "inline":
                keep = valid & (il_idx >= start) & (il_idx < stop)
            elif axis == "crossline":
                keep = valid & (xl_idx >= start) & (xl_idx < stop)
            else:
                keep = valid
            if not np.any(keep):
                continue
            picked = np.nonzero(keep)[0]
            traces = source.read_indices(a + picked)
            il_k, xl_k = il_idx[picked], xl_idx[picked]
            if axis == "inline":
                out[il_k - start, xl_k] = traces
            elif axis == "crossline":
                out[il_k, xl_k - start] = traces
            else:
                out[il_k, xl_k] = traces[:, start:stop]
    log.info(
        "Read %s amplitude window [%d, %d) of %s: %s in %.1fs",
        axis,
        start,
        stop,
        source.path.name,
        sysinfo.gb(out.nbytes),
        time.perf_counter() - started,
    )
    return out


# ---- large-file volumes -------------------------------------------------------


class _LargeVolume(SeismicVolume):
    large = True

    def __init__(self, name: str, path: Path, state: BuildState):
        self.name = name
        self.path = Path(path)
        self.format = self.path.suffix.lower().lstrip(".")
        self.state = state
        self._source: SeismicSource | None = None
        self._source_lock = threading.Lock()

    def status(self) -> dict | None:
        return self.state.to_json()

    def _open_source(self) -> SeismicSource:
        with self._source_lock:
            if self._source is None:
                self._source = open_source(self.path, self.state.layout)
            return self._source

    def _cellmap(self) -> np.ndarray | None:
        path = self.state.cellmap_path
        if path is None and self.state.manifest and self.state.manifest.get("has_cellmap"):
            path = str(self.state.directory / "cellmap.npy")
        if path is None:
            return None
        return np.load(path, mmap_mode="r")

    def amplitude_window(self, axis: str, start: int, stop: int) -> np.ndarray:
        layout = self.state.layout
        if layout is None or layout.kind != "3d":
            raise AxisNotReady("Volume layout is not known yet", self.status())
        return read_subcube(self._open_source(), layout, self._cellmap(), axis, start, stop)

    def close(self) -> None:
        with self._source_lock:
            if self._source is not None:
                self._source.close()
                self._source = None


class CachedVolume(_LargeVolume):
    """A large file served from its completed uint8 memmap cache."""

    def __init__(self, name: str, path: Path, state: BuildState):
        super().__init__(name, path, state)
        manifest = state.manifest
        assert manifest is not None
        self.manifest = manifest
        self.memmaps = open_memmaps(state.directory, manifest)
        self.geometry = load_geometry(state.directory / "geometry.npz")
        self.kind = manifest["kind"]
        self.shape = tuple(int(s) for s in manifest["shape"])

    def _plane(self, axis: str, index: int) -> np.ndarray:
        """The cube-order plane: inline (n_xl, n_s), crossline (n_il, n_s), time (n_il, n_xl)."""
        mm = self.memmaps
        if axis in mm:
            return np.asarray(mm[axis][index])
        if axis == "inline":
            if "crossline" in mm:
                return np.asarray(mm["crossline"][:, index, :])
            return np.asarray(mm["time"][:, index, :]).T
        if axis == "crossline":
            if "inline" in mm:
                return np.asarray(mm["inline"][:, index, :])
            return np.asarray(mm["time"][:, :, index]).T
        if "inline" in mm:
            return np.asarray(mm["inline"][:, :, index])
        return np.asarray(mm["crossline"][:, :, index]).T

    def rgb(self, axis: str, index: int) -> np.ndarray:
        if self.kind == "2d":
            section = self.memmaps["section"]
            pages = self.pages
            a, b = pages.bounds(index) if pages else (0, section.shape[1])
            return to_rgb(np.asarray(section[:, a:b]))
        return _rgb_from_planes(axis, index, self.axis_count(axis), self._plane)


class DirectVolume(_LargeVolume):
    """A large file whose cache is still building: storage-axis slices only."""

    _PLANE_CACHE = 12

    def __init__(self, name: str, path: Path, state: BuildState, listing: tuple[tuple[int, ...], SectionGeometry]):
        super().__init__(name, path, state)
        self._listing_shape, self._listing_geometry = listing
        self._exact_geometry: SectionGeometry | None = None
        self._planes: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._plane_lock = threading.Lock()

    @property
    def geometry(self) -> SectionGeometry:
        """Listing geometry until the exact scan lands, then the full geometry."""
        layout = self.state.layout
        if self._exact_geometry is None and layout is not None and not layout.provisional:
            try:
                source = self._open_source()
                source.layout = layout
                self._exact_geometry = source.geometry(self._cellmap())
            except Exception:
                log.exception("Could not build exact geometry for %s", self.name)
                return self._listing_geometry
        return self._exact_geometry or self._listing_geometry

    # The exact header scan can refine the grid after the volume is created.
    @property
    def kind(self) -> str:
        layout = self.state.layout
        return layout.kind if layout is not None else self._listing_geometry.kind

    @property
    def shape(self) -> tuple[int, ...]:
        layout = self.state.layout
        shape = layout.shape if layout is not None else self._listing_shape
        return tuple(int(s) for s in shape)

    @property
    def storage_axis(self) -> str | None:
        layout = self.state.layout
        if layout is None:
            return None
        if layout.kind != "3d":
            return "inline"
        return "inline" if layout.sort == "inline" else "crossline"

    def _direct_ready(self) -> bool:
        layout = self.state.layout
        if layout is None or self.state.lo is None or self.state.hi is None:
            return False
        if layout.kind != "3d" or layout.regular:
            return True
        return self.state.cellmap_path is not None

    def axes_ready(self) -> dict[str, bool]:
        ready = {axis: False for axis in AXES}
        if self._direct_ready():
            if self.kind == "2d":
                return {axis: True for axis in AXES}
            ready[self.storage_axis or "inline"] = True
        return ready

    def _not_ready(self, axis: str) -> AxisNotReady:
        status = self.status() or {}
        if not self._direct_ready():
            what = f"{status.get('stage', 'preparing')} ({status.get('percent', 0):.0f}%)"
            return AxisNotReady(
                f"{self.name} is still being prepared: {what}. Slices appear in a few seconds.",
                status,
            )
        return AxisNotReady(
            f"The {axis} axis of {self.name} becomes available when its cache finishes "
            f"({status.get('percent', 0):.0f}% done). The {self.storage_axis} axis works now.",
            status,
        )

    def _normalize(self, block: np.ndarray) -> np.ndarray:
        return normalize_block(block, float(self.state.lo), float(self.state.hi))

    def _storage_plane(self, index: int) -> np.ndarray:
        """Storage-axis plane in cube order, read straight from the source file."""
        key = (self.storage_axis or "inline", index)
        with self._plane_lock:
            cached = self._planes.get(key)
            if cached is not None:
                self._planes.move_to_end(key)
                return cached
        layout = self.state.layout
        assert layout is not None
        source = self._open_source()
        n_il, n_xl, n_s = layout.shape
        row = n_xl if layout.sort == "inline" else n_il
        if layout.regular:
            plane = self._normalize(source.read(index * row, (index + 1) * row))
        else:
            cellmap = self._cellmap()
            assert cellmap is not None
            il_idx, xl_idx = np.divmod(np.asarray(cellmap), n_xl)
            slow, fast = (il_idx, xl_idx) if layout.sort == "inline" else (xl_idx, il_idx)
            traces = np.nonzero((np.asarray(cellmap) >= 0) & (slow == index))[0]
            plane = np.full((row, n_s), self.state.fill_u8, dtype=np.uint8)
            if traces.size:
                plane[fast[traces]] = self._normalize(source.read_indices(traces))
        with self._plane_lock:
            self._planes[key] = plane
            while len(self._planes) > self._PLANE_CACHE:
                self._planes.popitem(last=False)
        return plane

    def rgb(self, axis: str, index: int) -> np.ndarray:
        if not self._direct_ready():
            raise self._not_ready(axis)
        if self.kind == "2d":
            pages = self.pages
            a, b = pages.bounds(index) if pages else (0, self.shape[1])
            block = self._open_source().read(a, b)
            return to_rgb(np.ascontiguousarray(self._normalize(block).T))
        if axis != self.storage_axis:
            raise self._not_ready(axis)
        return _rgb_from_planes(
            axis, index, self.axis_count(axis), lambda _axis, i: self._storage_plane(i)
        )
