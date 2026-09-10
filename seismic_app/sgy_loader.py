"""Step 1 of the pipeline: read seismic arrays while keeping geometry.

SEG-Y files are auto-detected as either 3D volumes (inline/crossline
headers populated and consistently sorted) or 2D lines. NumPy ``.npy``
files have no headers, so their dimensions define their geometry:

- 2D: ``(n_samples, n_traces)``
- 3D: ``(n_inlines, n_crosslines, n_samples)``

Singleton dimensions are removed before this check. NumPy geometry uses
index-valued inline/crossline coordinates and conservative default
spacings (25 m bins and a 4 ms sample interval).

Orientation convention: 2D sections are returned as (n_samples, n_traces)
- time increases down the rows, traces run along the columns - so the
array *is* the standard seismic display and SAM sees horizons as
horizontal features. segyio's trace accessor yields (n_traces, n_samples),
hence the transpose here, once, at the boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio

from .geometry import SectionGeometry, extract_2d_geometry, extract_3d_geometry
from .logutil import get_logger

log = get_logger("sgy_loader")

SUPPORTED_SEISMIC_SUFFIXES = frozenset({".sgy", ".npy"})
DEFAULT_NPY_SPACING_M = 25.0
DEFAULT_NPY_DT_MS = 4.0

# Header byte positions that may carry inline/crossline numbering. The
# SEG-Y standard says bytes 189/193, but many vendor exports put the grid
# elsewhere (e.g. SEGY0000.sgy stores inline in EnergySourcePoint byte 17
# and crossline in CDP_TRACE byte 25). Tried in order; first pair that
# yields a consistent sorted grid wins.
_ILINE_XLINE_CANDIDATES: list[tuple[int, int]] = [
    (int(segyio.TraceField.INLINE_3D), int(segyio.TraceField.CROSSLINE_3D)),  # 189/193
    (int(segyio.TraceField.EnergySourcePoint), int(segyio.TraceField.CDP_TRACE)),  # 17/25
    (int(segyio.TraceField.FieldRecord), int(segyio.TraceField.TraceNumber)),  # 9/13
    (int(segyio.TraceField.FieldRecord), int(segyio.TraceField.CDP)),  # 9/21
]


def _open_3d(path: str | Path):
    """Try to open a file as a sorted 3D volume, scanning header layouts.

    Returns an open segyio file (caller must close) or None if no
    candidate inline/crossline byte pair produces a valid grid.
    """
    for il_byte, xl_byte in _ILINE_XLINE_CANDIDATES:
        try:
            f = segyio.open(str(path), iline=il_byte, xline=xl_byte)
        except (RuntimeError, ValueError):
            continue
        try:
            n_il, n_xl = len(f.ilines), len(f.xlines)
        except (RuntimeError, ValueError):
            f.close()
            continue
        if n_il > 1 and n_xl > 1 and n_il * n_xl == f.tracecount:
            log.info(
                "Opened %s as 3D volume (%d inlines x %d crosslines) using "
                "header bytes iline=%d, xline=%d",
                path,
                n_il,
                n_xl,
                il_byte,
                xl_byte,
            )
            return f
        f.close()
    return None


def is_3d_volume(path: str | Path) -> bool:
    """True when the file can be opened as a sorted 3D volume."""
    f = _open_3d(path)
    if f is None:
        return False
    f.close()
    return True


def load_section(path: str | Path) -> tuple[np.ndarray, SectionGeometry]:
    """Read a 2D .sgy line and return (section, geometry).

    Returns
    -------
    section : np.ndarray
        (n_samples, n_traces) float32 - time down, traces across.
    geometry : SectionGeometry
        Physical axes read from the headers (dt, delay time, CDP numbers,
        world coordinates, trace spacing).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {path}")

    with segyio.open(str(path), ignore_geometry=True) as f:
        data = segyio.collect(f.trace[:])  # (n_traces, n_samples)
        geometry = extract_2d_geometry(f)

    section = np.ascontiguousarray(np.asarray(data, dtype=np.float32).T)
    return section, geometry


def load_volume(path: str | Path) -> tuple[np.ndarray, SectionGeometry]:
    """Read a 3D .sgy volume and return (cube, geometry).

    Returns
    -------
    cube : np.ndarray
        (n_ilines, n_xlines, n_samples) float32, segyio.tools.cube order.
    geometry : SectionGeometry
        kind="3d", with inline/crossline numbering and bin spacings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {path}")

    f = _open_3d(path)
    if f is None:
        raise RuntimeError(
            f"{path} could not be opened as a sorted 3D volume with any "
            "known inline/crossline header layout."
        )
    try:
        cube = np.asarray(segyio.tools.cube(f), dtype=np.float32)
        geometry = extract_3d_geometry(f)
    finally:
        f.close()

    return cube, geometry


def _numpy_shape(path: Path) -> tuple[int, ...]:
    """Read and validate an NPY shape without materializing its amplitudes."""
    try:
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError as exc:
        raise ValueError(f"{path.name} is not a numeric .npy array: {exc}") from exc
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise ValueError(
            f"{path.name} must contain a real numeric array, got dtype {raw.dtype}"
        )
    shape = tuple(size for size in raw.shape if size != 1)
    if len(shape) not in (2, 3):
        raise ValueError(
            f"{path.name} must be 2D or 3D after removing singleton dimensions; "
            f"got shape {raw.shape}"
        )
    if any(size == 0 for size in shape):
        raise ValueError(f"{path.name} contains an empty dimension: {shape}")
    return shape


def _numpy_geometry(shape: tuple[int, ...]) -> SectionGeometry:
    """Construct index-based physical geometry for a validated NPY shape."""
    if len(shape) == 2:
        n_samples, n_traces = shape
        trace_positions = np.arange(n_traces, dtype=np.float64)
        return SectionGeometry(
            kind="2d",
            n_traces=n_traces,
            n_samples=n_samples,
            dt_ms=DEFAULT_NPY_DT_MS,
            t0_ms=0.0,
            cdp=np.arange(n_traces, dtype=np.int64),
            world_x=trace_positions * DEFAULT_NPY_SPACING_M,
            world_y=np.zeros(n_traces, dtype=np.float64),
            distance_m=trace_positions * DEFAULT_NPY_SPACING_M,
            trace_spacing_m=DEFAULT_NPY_SPACING_M,
        )

    n_ilines, n_xlines, n_samples = shape
    return SectionGeometry(
        kind="3d",
        n_traces=n_ilines * n_xlines,
        n_samples=n_samples,
        dt_ms=DEFAULT_NPY_DT_MS,
        t0_ms=0.0,
        ilines=np.arange(n_ilines, dtype=np.int64),
        xlines=np.arange(n_xlines, dtype=np.int64),
        iline_spacing_m=DEFAULT_NPY_SPACING_M,
        xline_spacing_m=DEFAULT_NPY_SPACING_M,
    )


def load_numpy(path: str | Path) -> tuple[np.ndarray, SectionGeometry]:
    """Read a headerless NumPy section or volume using the app conventions."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NumPy file not found: {path}")

    try:
        raw = np.load(path, allow_pickle=False)
    except ValueError as exc:
        raise ValueError(f"{path.name} is not a numeric .npy array: {exc}") from exc

    shape = _numpy_shape(path)
    data = np.squeeze(raw)
    data = np.ascontiguousarray(data, dtype=np.float32)
    if not np.all(np.isfinite(data)):
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            raise ValueError(f"{path.name} contains no finite amplitude values")
        log.warning("Replacing non-finite amplitudes in %s with zero", path)
        data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    geometry = _numpy_geometry(shape)

    log.info("Loaded NumPy %s array %s using index geometry", geometry.kind, data.shape)
    return data, geometry


def inspect_any(path: str | Path) -> tuple[tuple[int, ...], SectionGeometry]:
    """Return shape and geometry without loading seismic amplitudes."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        shape = _numpy_shape(path)
        return shape, _numpy_geometry(shape)
    if suffix != ".sgy":
        supported = ", ".join(sorted(SUPPORTED_SEISMIC_SUFFIXES))
        raise ValueError(f"Unsupported seismic file type {suffix!r}; expected {supported}")

    f = _open_3d(path)
    if f is not None:
        try:
            shape = (len(f.ilines), len(f.xlines), len(f.samples))
            return shape, extract_3d_geometry(f)
        finally:
            f.close()
    with segyio.open(str(path), ignore_geometry=True) as f:
        shape = (len(f.samples), f.tracecount)
        return shape, extract_2d_geometry(f)


def load_any(path: str | Path) -> tuple[np.ndarray, SectionGeometry]:
    """Load a supported seismic file and return app-oriented data and geometry.

    2D lines come back as (n_samples, n_traces) sections; 3D volumes as
    (n_ilines, n_xlines, n_samples) cubes. Check geometry.kind to tell
    them apart.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return load_numpy(path)
    if suffix != ".sgy":
        supported = ", ".join(sorted(SUPPORTED_SEISMIC_SUFFIXES))
        raise ValueError(f"Unsupported seismic file type {suffix!r}; expected {supported}")
    f = _open_3d(path)
    if f is not None:
        try:
            cube = np.asarray(segyio.tools.cube(f), dtype=np.float32)
            geometry = extract_3d_geometry(f)
        finally:
            f.close()
        return cube, geometry
    log.info("Loading %s as a 2D line (no 3D grid found in headers)", path)
    return load_section(path)
