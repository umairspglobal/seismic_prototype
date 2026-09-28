"""Uniform streaming access to seismic files that may not fit in RAM.

Both SEG-Y and NumPy volumes are exposed as a sequence of traces in
file order plus a ``GridLayout`` saying where each trace sits, so the
cache builder and the direct slice reader never need to know which
format they are reading.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from .geometry import SectionGeometry
from .logutil import get_logger
from .segy_geometry import (
    GridLayout,
    TraceReader,
    build_geometry,
    detect_layout_fast,
    listing_geometry,
)

log = get_logger("sources")


class SeismicSource:
    """Traces in file order plus the layout that places them on a grid."""

    path: Path
    layout: GridLayout
    format: str

    @property
    def n_traces(self) -> int:
        return self.layout.n_traces

    @property
    def n_samples(self) -> int:
        return self.layout.n_samples

    @property
    def n_values(self) -> int:
        return self.n_traces * self.n_samples

    def read(self, start: int, stop: int) -> np.ndarray:
        """Traces [start, stop) as (n, n_samples) float32."""
        raise NotImplementedError

    def read_indices(self, indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample_traces(self, count: int = 20_000) -> np.ndarray:
        """About ``count`` traces spread evenly through the file, as float32."""
        count = max(1, min(count, self.n_traces))
        indices = np.unique(np.linspace(0, self.n_traces - 1, count).astype(np.int64))
        return self.read_indices(indices)

    def geometry(self, cellmap: np.ndarray | None = None) -> SectionGeometry:
        raise NotImplementedError

    def listing_geometry(self) -> SectionGeometry:
        return self.geometry()

    def close(self) -> None:
        pass

    def __enter__(self) -> "SeismicSource":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class SegySource(SeismicSource):
    format = "sgy"

    def __init__(self, path: str | Path, layout: GridLayout | None = None):
        self.path = Path(path)
        self.reader = TraceReader(self.path)
        self.layout = layout or detect_layout_fast(self.reader, self.path)

    def read(self, start: int, stop: int) -> np.ndarray:
        return self.reader.read(start, stop)

    def read_indices(self, indices: np.ndarray) -> np.ndarray:
        return self.reader.read_indices(indices)

    def geometry(self, cellmap: np.ndarray | None = None) -> SectionGeometry:
        return build_geometry(self.reader, self.layout, cellmap)

    def listing_geometry(self) -> SectionGeometry:
        return listing_geometry(self.reader, self.layout)

    def close(self) -> None:
        self.reader.close()


class NpySource(SeismicSource):
    """Memory-mapped .npy: 2D (n_samples, n_traces) or 3D (n_il, n_xl, n_samples)."""

    format = "npy"

    def __init__(self, path: str | Path):
        from .sgy_loader import _numpy_shape

        self.path = Path(path)
        shape = _numpy_shape(self.path)
        raw = np.load(self.path, mmap_mode="r", allow_pickle=False)
        self.array = raw.reshape(shape)
        self._warned_nonfinite = False
        self._lock = threading.Lock()
        if len(shape) == 3:
            n_il, n_xl, n_samples = shape
            self._traces = self.array.reshape(n_il * n_xl, n_samples)
            self.layout = GridLayout(
                kind="3d",
                n_traces=n_il * n_xl,
                n_samples=n_samples,
                sort="inline",
                n_il=n_il,
                n_xl=n_xl,
                regular=True,
                provisional=False,
            )
        else:
            n_samples, n_traces = shape
            self._traces = None
            self.layout = GridLayout(
                kind="2d", n_traces=n_traces, n_samples=n_samples, provisional=False
            )

    def _finite(self, block: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(block)):
            if not self._warned_nonfinite:
                log.warning("Replacing non-finite amplitudes in %s with zero", self.path.name)
                self._warned_nonfinite = True
            np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return block

    def read(self, start: int, stop: int) -> np.ndarray:
        stop = min(stop, self.n_traces)
        if self._traces is not None:
            block = np.array(self._traces[start:stop], dtype=np.float32)
        else:
            block = np.array(self.array[:, start:stop].T, dtype=np.float32)
        return self._finite(block)

    def read_indices(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if self._traces is not None:
            block = np.array(self._traces[indices], dtype=np.float32)
        else:
            block = np.array(self.array[:, indices].T, dtype=np.float32)
        return self._finite(block)

    def geometry(self, cellmap: np.ndarray | None = None) -> SectionGeometry:
        from .sgy_loader import _numpy_geometry

        return _numpy_geometry(self.layout.shape if self.layout.kind == "3d" else self.array.shape)

    def close(self) -> None:
        self.array = None
        self._traces = None


def open_source(path: str | Path, layout: GridLayout | None = None) -> SeismicSource:
    path = Path(path)
    started = time.perf_counter()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        source: SeismicSource = NpySource(path)
    elif suffix == ".sgy":
        source = SegySource(path, layout)
    else:
        raise ValueError(f"Unsupported seismic file type {suffix!r}")
    log.debug("Opened %s source in %.3fs", path.name, time.perf_counter() - started)
    return source
