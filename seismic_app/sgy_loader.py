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


def is_3d_volume(path: str | Path) -> bool:
    """True when segyio can open the file as a sorted 3D volume."""
    try:
        with segyio.open(str(path), ignore_geometry=False):
            return True
    except (RuntimeError, ValueError):
        return False


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

    with segyio.open(str(path), ignore_geometry=False) as f:
        cube = np.asarray(segyio.tools.cube(f), dtype=np.float32)
        geometry = extract_3d_geometry(f)

    return cube, geometry


def load_any(path: str | Path) -> tuple[np.ndarray, SectionGeometry]:
    """Auto-detect 2D vs 3D and load accordingly.

    2D lines come back as (n_samples, n_traces) sections; 3D volumes as
    (n_ilines, n_xlines, n_samples) cubes. Check geometry.kind to tell
    them apart.
    """
    if is_3d_volume(path):
        return load_volume(path)
    return load_section(path)
