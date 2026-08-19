import numpy as np

from app import (
    display_to_slice_coordinates,
    materialize_interactive_mask,
    saved_mask_for_slice,
)


def test_display_coordinates_are_scaled_and_clamped():
    assert display_to_slice_coordinates(250, 500, (500, 1000), (100, 200)) == (
        50,
        100,
    )
    assert display_to_slice_coordinates(-20, 5000, (500, 1000), (100, 200)) == (
        0,
        199,
    )


def test_lazy_inline_slice_materializes_only_at_export():
    mask = np.zeros((4, 3), dtype=bool)
    mask[2, 1] = True
    saved = {
        "kind": "slice",
        "mask": mask,
        "axis": "inline",
        "index": 1,
        "volume_shape": (2, 4, 3),
    }

    assert saved_mask_for_slice(saved, "inline", 0) is None
    assert saved_mask_for_slice(saved, "inline", 1) is mask
    volume = materialize_interactive_mask(saved)
    assert volume.shape == (2, 4, 3)
    assert volume.sum() == 1
    assert volume[1, 2, 1]


def test_frame_stack_uses_transposed_volume_view_for_crosslines():
    # (crossline, sample, inline)
    frames = np.zeros((3, 4, 2), dtype=bool)
    frames[2, 1, 0] = True
    saved = {
        "kind": "frames",
        "masks": frames,
        "axis": "crossline",
        "volume_shape": (2, 4, 3),
    }

    volume = materialize_interactive_mask(saved)
    assert volume.shape == (2, 4, 3)
    assert np.shares_memory(volume, frames)
    assert volume[0, 1, 2]
    inline = saved_mask_for_slice(saved, "inline", 0)
    assert inline.shape == (4, 3)
    assert inline[1, 2]
