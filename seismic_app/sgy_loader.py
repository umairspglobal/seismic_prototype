"""Step 1 of the pipeline: read .sgy files with segyio, keeping geometry.

Files are auto-detected as either 3D volumes (inline/crossline headers
populated and consistently sorted) or 2D lines (everything else - all the
current files in data/ are 2D crooked lines with CDP numbering and
per-trace navigation coordinates).

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


def load_any(path: str | Path) -> tuple[np.ndarray, SectionGeometry]:
    """Auto-detect 2D vs 3D and load accordingly.

    2D lines come back as (n_samples, n_traces) sections; 3D volumes as
    (n_ilines, n_xlines, n_samples) cubes. Check geometry.kind to tell
    them apart.
    """
    if is_3d_volume(path):
        return load_volume(path)
    log.info("Loading %s as a 2D line (no 3D grid found in headers)", path)
    return load_section(path)
