import numpy as np
import pytest

import server.main as api
from seismic_app.sgy_loader import load_any
from seismic_app.vtk_export import export_volume_vti


def test_loads_3d_npy_with_volume_geometry(tmp_path):
    expected = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    path = tmp_path / "cube.npy"
    np.save(path, expected)

    data, geometry = load_any(path)

    assert data.dtype == np.float32
    assert data.flags.c_contiguous
    assert np.array_equal(data, expected)
    assert geometry.kind == "3d"
    assert geometry.n_traces == 6
    assert geometry.n_samples == 4
    assert np.array_equal(geometry.ilines, np.arange(2))
    assert np.array_equal(geometry.xlines, np.arange(3))


def test_loads_2d_npy_in_display_orientation(tmp_path):
    expected = np.arange(15, dtype=np.int16).reshape(5, 3)
    path = tmp_path / "line.npy"
    np.save(path, expected)

    data, geometry = load_any(path)

    assert np.array_equal(data, expected)
    assert geometry.kind == "2d"
    assert geometry.n_samples == 5
    assert geometry.n_traces == 3


def test_npy_rejects_unsupported_shape_and_object_arrays(tmp_path):
    four_d = tmp_path / "four_d.npy"
    objects = tmp_path / "objects.npy"
    np.save(four_d, np.zeros((2, 2, 2, 2), dtype=np.float32))
    np.save(objects, np.array([{"amplitude": 1}], dtype=object))

    with pytest.raises(ValueError, match="must be 2D or 3D"):
        load_any(four_d)
    with pytest.raises(ValueError, match="not a numeric .npy array"):
        load_any(objects)


def test_api_lists_npy_files_with_axes(tmp_path, monkeypatch):
    np.save(tmp_path / "volume.npy", np.zeros((2, 3, 4), dtype=np.float32))
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    api._file_cache.clear()

    files = api.list_files(format="npy")

    assert files == [
        {
            "name": "volume.npy",
            "format": "npy",
            "kind": "3d",
            "shape": [2, 3, 4],
            "axes": {"inline": 2, "crossline": 3, "time": 4},
        }
    ]
    # Discovery reads only the NPY header; amplitudes are loaded on selection.
    assert api._file_cache == {}


def test_npy_geometry_exports_to_vti(tmp_path):
    path = tmp_path / "volume.npy"
    np.save(path, np.zeros((2, 3, 4), dtype=np.float32))
    data, geometry = load_any(path)
    masks = {"object 1": np.zeros((2, 4, 3), dtype=bool)}
    masks["object 1"][1, 2, 1] = True

    written = export_volume_vti(masks, data, geometry, tmp_path / "npy_export")

    assert written.is_file()
    assert written.suffix == ".vti"
