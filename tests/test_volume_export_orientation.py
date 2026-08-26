"""The exported label cube must land on the samples the user clicked.

Propagated masks are per-slice images whose orientation differs per
axis (inlines and crosslines are transposed for display, time slices are
not). If the inverse mapping is wrong the ParaView export is silently
shifted or transposed, which is hard to spot by eye, so it is pinned
here against the real display path.
"""

import numpy as np

from seismic_app.geometry import SectionGeometry
from seismic_app.vtk_export import export_volume_vti
from server.main import _slice_masks_to_cube, _slice_rgb

N_IL, N_XL, N_SAMPLES = 3, 5, 7


def _geometry() -> SectionGeometry:
    return SectionGeometry(
        kind="3d",
        n_traces=N_IL * N_XL,
        n_samples=N_SAMPLES,
        dt_ms=4.0,
        t0_ms=0.0,
        ilines=np.arange(N_IL),
        xlines=np.arange(N_XL),
        iline_spacing_m=25.0,
        xline_spacing_m=12.5,
    )


def _cube() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(N_IL, N_XL, N_SAMPLES), dtype=np.uint8)


def test_slice_masks_invert_back_onto_the_source_samples():
    data_u8 = _cube()
    geometry = _geometry()
    expected = data_u8 > 127

    for axis, n_frames in (
        ("inline", N_IL),
        ("crossline", N_XL),
        ("time", N_SAMPLES),
    ):
        # Threshold every slice as it is displayed, then fold the masks
        # back into volume space; the result must match thresholding the
        # cube directly.
        frames = [
            _slice_rgb(data_u8, geometry, axis, i)[:, :, 1] > 127
            for i in range(n_frames)
        ]
        cube = _slice_masks_to_cube(
            np.asarray(frames), axis, (N_IL, N_XL, N_SAMPLES)
        )
        assert cube.shape == expected.shape
        assert np.array_equal(cube, expected), f"{axis} masks are misoriented"


def test_export_writes_a_readable_vti(tmp_path):
    masks = np.zeros((N_IL, N_XL, N_SAMPLES), dtype=bool)
    masks[1, 2, 3] = True
    # vtk_export takes (n_il, n_samples, n_xl) stacks.
    written = export_volume_vti(
        {"object 1": np.transpose(masks, (0, 2, 1))},
        _cube().astype(np.float32),
        _geometry(),
        tmp_path / "volume",
    )
    assert written.exists() and written.suffix == ".vti"
    assert written.stat().st_size > 0


def test_export_without_amplitude_is_smaller():
    import tempfile
    from pathlib import Path

    masks = {"object 1": np.zeros((N_IL, N_SAMPLES, N_XL), dtype=bool)}
    amplitude = _cube().astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        both = export_volume_vti(
            masks, amplitude, _geometry(), Path(tmp) / "both"
        ).stat().st_size
        labels_only = export_volume_vti(
            masks, amplitude, _geometry(), Path(tmp) / "labels", include_amplitude=False
        ).stat().st_size
    assert labels_only < both
