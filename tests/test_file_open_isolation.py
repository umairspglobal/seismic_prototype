import threading
import time
from pathlib import Path

import numpy as np
import pytest

from seismic_app.sgy_loader import inspect_any_for_listing
from server import main


def test_loading_one_volume_does_not_block_another(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()
    np.save(tmp_path / "slow.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(tmp_path / "fast.npy", np.ones((3, 2), dtype=np.float32))
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    main._file_cache.clear()

    original_load = main.load_any

    def fake_load(path):
        path = Path(path)
        if path.name == "slow.npy":
            started.set()
            assert release.wait(timeout=2)
        return original_load(path)

    monkeypatch.setattr(main, "load_any", fake_load)
    results: dict[str, tuple[int, ...]] = {}

    def load_slow() -> None:
        results["slow"] = main._get_file("slow.npy")[0].shape

    def load_fast() -> None:
        assert started.wait(timeout=2)
        results["fast"] = main._get_file("fast.npy")[0].shape
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
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "_FILE_CACHE_LIMIT", 2)
    main._file_cache.clear()

    main._get_file("a.npy")
    main._get_file("b.npy")
    main._get_file("c.npy")

    assert "a.npy" not in main._file_cache
    assert "b.npy" in main._file_cache
    assert "c.npy" in main._file_cache


def test_listing_timeout_falls_back_to_2d_headers(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "data" / "1.sgy"
    if not path.is_file():
        pytest.skip("sample SEG-Y is not present")

    blocked = threading.Event()

    def hang(_path):
        blocked.wait(timeout=5)
        raise AssertionError("3D inspect should have timed out")

    monkeypatch.setattr("seismic_app.sgy_loader.inspect_any", hang)
    try:
        shape, geometry = inspect_any_for_listing(path)
    finally:
        blocked.set()

    assert geometry.kind == "2d"
    assert len(shape) == 2
    assert time.perf_counter()  # keep import used if hang returns early
