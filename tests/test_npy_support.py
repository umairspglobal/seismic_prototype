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


def test_loading_one_volume_does_not_block_another(monkeypatch, tmp_path):
    import threading
    from pathlib import Path

    started = threading.Event()
    release = threading.Event()
    np.save(tmp_path / "slow.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(tmp_path / "fast.npy", np.ones((3, 2), dtype=np.float32))
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    api._file_cache.clear()

    original_load = api.load_any

    def fake_load(path):
        path = Path(path)
        if path.name == "slow.npy":
            started.set()
            assert release.wait(timeout=2)
        return original_load(path)

    monkeypatch.setattr(api, "load_any", fake_load)
    results = {}

    def load_slow():
        results["slow"] = api._get_file("slow.npy")[0].shape

    def load_fast():
        assert started.wait(timeout=2)
        results["fast"] = api._get_file("fast.npy")[0].shape
        release.set()

    slow_thread = threading.Thread(target=load_slow)
    fast_thread = threading.Thread(target=load_fast)
    slow_thread.start()
    fast_thread.start()
    slow_thread.join(timeout=3)
    fast_thread.join(timeout=3)

    assert results["fast"] == (3, 2)
    assert results["slow"] == (2, 2)


def test_file_cache_evicts_oldest_survey(monkeypatch, tmp_path):
    for name, shape in (("a.npy", (2, 2)), ("b.npy", (3, 2)), ("c.npy", (4, 2))):
        np.save(tmp_path / name, np.zeros(shape, dtype=np.float32))
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api, "_FILE_CACHE_LIMIT", 2)
    api._file_cache.clear()

    api._get_file("a.npy")
    api._get_file("b.npy")
    api._get_file("c.npy")

    assert "a.npy" not in api._file_cache
    assert "b.npy" in api._file_cache
    assert "c.npy" in api._file_cache


def test_listing_timeout_falls_back_to_2d_headers(monkeypatch):
    from pathlib import Path

    from seismic_app.sgy_loader import inspect_any_for_listing

    path = Path(__file__).resolve().parents[1] / "data" / "1.sgy"
    if not path.is_file():
        pytest.skip("sample SEG-Y is not present")

    def boom(_path):
        raise AssertionError("listing must not build a 3D SEG-Y index")

    monkeypatch.setattr("seismic_app.sgy_loader._open_3d", boom)
    shape, geometry = inspect_any_for_listing(path)

    assert geometry.kind == "2d"
    assert len(shape) == 2
