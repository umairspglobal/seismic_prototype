"""ParaView export: write SAM 3 masks with the *real* SEG-Y geometry.

Two products per 2D line:

- ``<name>.vts`` (StructuredGrid) - the primary export. Every trace is
  placed at its world (x, y) from the SEG-Y headers with time as
  negative-down z, mirroring what ParaView's own SEG-Y reader builds for
  2D lines, so the masks land exactly on the seismic. The raw amplitude
  is included alongside the label volume so alignment can be verified
  even without loading the .sgy itself.
- ``<name>.vti`` (ImageData) - a secondary flat panel in
  (distance-along-line, time) space for quick inspection.

3D volumes export a single ``<name>.vti`` with inline/crossline bin
spacings and the sample interval as the axis spacings.

All five feature masks are encoded into one integer label array
(0=background, 1=fault, 2=channel, 3=facies, 4=salt, 5=horizon) for the
ParaView Threshold-per-label workflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .geometry import SectionGeometry

# fault=1, channel=2, seismic facies=3, salt body=4, horizon=5 - this order
# matches config.LABEL_STYLES and the ParaView label/color table.
LABEL_IDS: dict[str, int] = {
    style.noun_phrase: i + 1 for i, style in enumerate(config.LABEL_STYLES)
}
# Interactive point-picked masks get IDs above the fixed vocabulary.
FIRST_INTERACTIVE_LABEL_ID = len(config.LABEL_STYLES) + 1


def masks_to_label_volume(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Combine per-label boolean masks into one int32 label array.

    0 = background. Labels in LABEL_IDS keep their fixed IDs; any other
    (e.g. interactively picked) masks get sequential IDs starting at
    FIRST_INTERACTIVE_LABEL_ID. Later masks win on overlap.
    """
    shape = next(iter(masks.values())).shape
    combined = np.zeros(shape, dtype=np.int32)
    next_free = FIRST_INTERACTIVE_LABEL_ID
    for name, mask in masks.items():
        label_id = LABEL_IDS.get(name)
        if label_id is None:
            label_id = next_free
            next_free += 1
        combined[mask] = label_id
    return combined


def _sample_depths(geometry: SectionGeometry) -> np.ndarray:
    """Negative-down z coordinate of every sample (ms of two-way time)."""
    return -(geometry.t0_ms + np.arange(geometry.n_samples) * geometry.dt_ms)


def export_section_vts(
    masks: dict[str, np.ndarray],
    amplitude: np.ndarray,
    geometry: SectionGeometry,
    out_path: str | Path,
) -> Path:
    """Write a 2D line as a StructuredGrid in world coordinates.

    masks/amplitude are (n_samples, n_traces) section arrays. The grid
    points are (world_x[i], world_y[i], -(t0 + j*dt)) - the same layout
    ParaView's SEG-Y reader produces for 2D lines, so the exported masks
    overlay the seismic exactly.
    """
    from pyevtk.hl import gridToVTK  # deferred: optional dependency

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_traces, n_samples = geometry.n_traces, geometry.n_samples
    z_depths = _sample_depths(geometry)

    # Node coordinate arrays, shape (n_traces, n_samples, 1).
    x = np.broadcast_to(geometry.world_x[:, None, None], (n_traces, n_samples, 1))
    y = np.broadcast_to(geometry.world_y[:, None, None], (n_traces, n_samples, 1))
    z = np.broadcast_to(z_depths[None, :, None], (n_traces, n_samples, 1))

    labels = masks_to_label_volume(masks)  # (n_samples, n_traces)

    point_data = {
        "label": np.ascontiguousarray(labels.T[:, :, None]),
        "amplitude": np.ascontiguousarray(
            amplitude.T[:, :, None].astype(np.float32)
        ),
    }

    written = gridToVTK(
        str(out_path.with_suffix("")),
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(z, dtype=np.float64),
        pointData=point_data,
    )
    return Path(written)


def export_section_vti(
    masks: dict[str, np.ndarray],
    amplitude: np.ndarray,
    geometry: SectionGeometry,
    out_path: str | Path,
) -> Path:
    """Write a 2D line as flat ImageData in (distance, time) space.

    x = distance along the line (median trace spacing), y = flat, z =
    negative-down time. Sample order is flipped along z so the VTI
    spacing stays positive while depth still increases downward.
    """
    from pyevtk.hl import imageToVTK  # deferred: optional dependency

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = masks_to_label_volume(masks)  # (n_samples, n_traces)
    z_depths = _sample_depths(geometry)

    # (n_traces, 1, n_samples), deepest sample first so z increases.
    def _to_vti(section: np.ndarray, dtype) -> np.ndarray:
        return np.ascontiguousarray(section.T[:, None, ::-1].astype(dtype))

    written = imageToVTK(
        str(out_path.with_suffix("")),
        origin=(0.0, 0.0, float(z_depths[-1])),
        spacing=(geometry.trace_spacing_m, 1.0, geometry.dt_ms),
        pointData={
            "label": _to_vti(labels, np.int32),
            "amplitude": _to_vti(amplitude, np.float32),
        },
    )
    return Path(written)


def export_volume_vti(
    masks: dict[str, np.ndarray],
    amplitude: np.ndarray,
    geometry: SectionGeometry,
    out_path: str | Path,
    include_amplitude: bool = True,
) -> Path:
    """Write a 3D volume as ImageData with real bin/sample spacings.

    masks are (n_ilines, n_samples, n_xlines) stacks (pipeline
    convention); amplitude is the raw (n_ilines, n_xlines, n_samples)
    cube. Axes: x=inline direction, y=crossline direction, z=time
    (negative down). The survey's world rotation is not encoded - apply
    a transform in ParaView if you need true world placement.

    Set include_amplitude=False to write labels only; the file is then
    roughly half the size, at the cost of not being able to check the
    mask against the seismic without loading the .sgy separately.
    """
    from pyevtk.hl import imageToVTK  # deferred: optional dependency

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = masks_to_label_volume(masks)  # (n_il, n_samples, n_xl)
    labels_cube = np.transpose(labels, (0, 2, 1))  # -> (n_il, n_xl, n_samples)
    z_depths = _sample_depths(geometry)

    point_data = {
        "label": np.ascontiguousarray(labels_cube[:, :, ::-1].astype(np.int32))
    }
    if include_amplitude:
        point_data["amplitude"] = np.ascontiguousarray(
            amplitude[:, :, ::-1].astype(np.float32)
        )

    written = imageToVTK(
        str(out_path.with_suffix("")),
        origin=(0.0, 0.0, float(z_depths[-1])),
        spacing=(
            geometry.iline_spacing_m or 25.0,
            geometry.xline_spacing_m or 25.0,
            geometry.dt_ms,
        ),
        pointData=point_data,
    )
    return Path(written)


def export_masks(
    masks: dict[str, np.ndarray],
    amplitude: np.ndarray,
    geometry: SectionGeometry,
    out_base: str | Path,
) -> list[Path]:
    """Geometry-aware export dispatch. Returns the paths written."""
    out_base = Path(out_base)
    if geometry.kind == "3d":
        return [export_volume_vti(masks, amplitude, geometry, out_base)]
    return [
        export_section_vts(masks, amplitude, geometry, out_base),
        export_section_vti(masks, amplitude, geometry, out_base),
    ]
