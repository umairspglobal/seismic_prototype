"""Step 1 of the pipeline: read 2D .sgy seismic sections with segyio.

Each file in data/ is treated as a single 2D section (inline or crossline).
segyio returns traces as a NumPy ndarray via its "virtual array" trace
accessor, and handles IBM/IEEE float conversion transparently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio


def load_2d_section(path: str | Path) -> np.ndarray:
    """Read a 2D .sgy file and return its amplitude section.

    Returns
    -------
    np.ndarray
        Array of shape (n_traces, n_samples), float32. Traces are ordered
        along the profile, so row 0 is the first trace on the line.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {path}")

    with segyio.open(str(path), ignore_geometry=True) as f:
        data = segyio.collect(f.trace[:])

    return np.asarray(data, dtype=np.float32)


def iter_inlines_3d(path: str | Path):
    """Yield each inline of a 3D .sgy volume as a 2D (n_xlines, n_samples) array.

    Use this instead of load_2d_section when a file is a full 3D volume
    rather than a single 2D line (see guide section 8: "if your files are
    3D volumes ... use segyio's cube() function and iterate over each
    inline slice").
    """
    path = Path(path)
    with segyio.open(str(path), ignore_geometry=False) as f:
        cube = segyio.tools.cube(f)  # (n_inlines, n_xlines, n_samples)
        for inline_idx in range(cube.shape[0]):
            yield np.asarray(cube[inline_idx], dtype=np.float32)
