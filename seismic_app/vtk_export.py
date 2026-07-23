"""Step 9 (ParaView export): write SAM 3 masks as VTK ImageData (.vti).

Encodes all five feature masks into a single integer label volume
(0=background, 1=fault, 2=channel, 3=facies, 4=salt, 5=horizon - matching
the guide's ParaView Threshold-per-label workflow) and writes it with
PyEVTK's imageToVTK. Open the resulting .vti directly in ParaView, or
Threshold it per label ID and color each per the guide's table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config

# fault=1, channel=2, seismic facies=3, salt body=4, horizon=5 - this order
# matches config.LABEL_STYLES and the guide's ParaView label/color table.
LABEL_IDS: dict[str, int] = {
    style.noun_phrase: i + 1 for i, style in enumerate(config.LABEL_STYLES)
}


def masks_to_label_volume(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Combine per-label boolean masks into one int32 label array.

    0 = background; otherwise LABEL_IDS[noun_phrase]. Later labels in
    config.LABEL_STYLES win on pixel overlap (rare, since masks are
    independent per-prompt binary maps).
    """
    shape = next(iter(masks.values())).shape
    combined = np.zeros(shape, dtype=np.int32)
    for noun_phrase, label_id in LABEL_IDS.items():
        mask = masks.get(noun_phrase)
        if mask is not None:
            combined[mask] = label_id
    return combined


def export_vti(
    masks: dict[str, np.ndarray],
    out_path: str | Path,
    trace_spacing: float = 25.0,
    sample_interval: float = 4.0,
) -> Path:
    """Write the combined label volume to a .vti file for ParaView.

    Parameters
    ----------
    trace_spacing: physical spacing between traces in metres (section's
        first axis - inline/crossline bin size).
    sample_interval: sample interval along the trace in ms (section's
        second axis). Per the guide's "coordinate alignment" tip, these
        must match the .sgy geometry loaded via ParaView's SegYReader or
        the mask will appear offset from the seismic.
    """
    from pyevtk.hl import imageToVTK  # deferred: optional dependency

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combined = masks_to_label_volume(masks)
    # pyevtk expects 3D arrays (nx, ny, nz); add a unit z-axis for a 2D section.
    combined_3d = np.ascontiguousarray(combined[:, :, np.newaxis])

    # imageToVTK appends ".vti" to the given path and returns the full path written.
    written = imageToVTK(
        str(out_path.with_suffix("")),
        cellData={"label": combined_3d},
        spacing=(trace_spacing, sample_interval, 1.0),
    )
    return Path(written)
