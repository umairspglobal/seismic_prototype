"""Large-file path: geometry detection, disk cache, direct reads, windowing.

Everything runs on small synthetic SEG-Y / NumPy files; the in-memory
budget is forced to zero where the server routing is exercised, so the
same code paths as a multi-GB survey are taken.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import segyio

from seismic_app import config
from seismic_app.preprocessing import inline_to_rgb_25d, normalize_to_uint8, to_rgb
from seismic_app.segy_geometry import TraceReader, detect_layout_fast, scan_layout_exact
from seismic_app.sgy_loader import inspect_any_for_listing, load_volume
from seismic_app.volume import AxisNotReady, CachedVolume, DirectVolume, make_pages, slice_rgb
from seismic_app.volume_cache import (
    BuildState,
    CacheManager,
    build_cache,
    cache_dir_for,
    normalize_block,
    open_memmaps,
    read_manifest,
)

N_IL, N_XL, N_S = 12, 17, 40


def _cube(seed: int = 0, shape=(N_IL, N_XL, N_S)) -> np.ndarray:
    # Multiples of 1/16 are exact in both IBM and IEEE floats.
    rng = np.random.default_rng(seed)
    return (np.round(rng.normal(0, 40, size=shape)) / 16).astype(np.float32)


def write_segy(
    path: Path,
    cube: np.ndarray,
    *,
    il_byte: int = 189,
    xl_byte: int = 193,
    sort: str = "inline",
    fmt: int = 1,
    il_start: int = 100,
    il_step: int = 2,
    xl_start: int = 1000,
    xl_step: int = 1,
    drop: set[tuple[int, int]] = frozenset(),
) -> Path:
    n_il, n_xl, n_s = cube.shape
    if sort == "inline":
        cells = [(i, j) for i in range(n_il) for j in range(n_xl)]
    else:
        cells = [(i, j) for j in range(n_xl) for i in range(n_il)]
    cells = [c for c in cells if c not in drop]
    spec = segyio.spec()
    spec.format = fmt
    spec.samples = list(range(n_s))
    spec.tracecount = len(cells)
    with segyio.create(str(path), spec) as f:
        f.bin.update({segyio.BinField.Samples: n_s, segyio.BinField.Interval: 4000, segyio.BinField.Format: fmt})
        for t, (i, j) in enumerate(cells):
            f.header[t] = {
                il_byte: il_start + i * il_step,
                xl_byte: xl_start + j * xl_step,
                segyio.TraceField.SourceGroupScalar: 1,
                segyio.TraceField.CDP_X: 500_000 + j * 25,
                segyio.TraceField.CDP_Y: 6_000_000 + i * 25 * il_step,
                segyio.TraceField.TRACE_SAMPLE_COUNT: n_s,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: 4000,
            }
            f.trace[t] = cube[i, j]
    return path


def write_segy_2d(path: Path, section: np.ndarray, fmt: int = 5) -> Path:
    """``section`` is (n_traces, n_samples); CDP numbers along the line."""
    n_tr, n_s = section.shape
    spec = segyio.spec()
    spec.format = fmt
    spec.samples = list(range(n_s))
    spec.tracecount = n_tr
    with segyio.create(str(path), spec) as f:
        f.bin.update({segyio.BinField.Samples: n_s, segyio.BinField.Interval: 2000, segyio.BinField.Format: fmt})
        for t in range(n_tr):
            f.header[t] = {
                segyio.TraceField.TRACE_SEQUENCE_LINE: t + 1,
                segyio.TraceField.CDP: 5000 + t,
                segyio.TraceField.SourceGroupScalar: 1,
                segyio.TraceField.CDP_X: 400_000 + t * 12,
                segyio.TraceField.CDP_Y: 7_000_000,
                segyio.TraceField.TRACE_SAMPLE_COUNT: n_s,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: 2000,
            }
            f.trace[t] = section[t]
    return path


def _expected_u8(cube: np.ndarray, manifest: dict, missing=()) -> np.ndarray:
    expected = normalize_block(cube.copy(), float(manifest["lo"]), float(manifest["hi"]))
    for i, j in missing:
        expected[i, j] = manifest["fill_u8"]
    return expected


def _assert_orders(directory: Path, manifest: dict, expected: np.ndarray) -> None:
    memmaps = open_memmaps(directory, manifest)
    assert set(manifest["orders"]) == {"inline", "crossline", "time"}
    np.testing.assert_array_equal(np.asarray(memmaps["inline"]), expected)
    np.testing.assert_array_equal(np.asarray(memmaps["crossline"]), expected.transpose(1, 0, 2))
    np.testing.assert_array_equal(np.asarray(memmaps["time"]), expected.transpose(2, 0, 1))


# ---- geometry detection --------------------------------------------------------


@pytest.mark.parametrize(
    "il_byte, xl_byte, sort, fmt",
    [
        (189, 193, "inline", 1),
        (17, 25, "inline", 5),
        (9, 21, "crossline", 1),
        (189, 193, "crossline", 5),
    ],
)
def test_layout_detection_and_cache_roundtrip(tmp_path, il_byte, xl_byte, sort, fmt):
    cube = _cube(1)
    path = write_segy(tmp_path / "survey.sgy", cube, il_byte=il_byte, xl_byte=xl_byte, sort=sort, fmt=fmt)

    reader = TraceReader(path)
    layout = detect_layout_fast(reader, path)
    assert (layout.il_byte, layout.xl_byte) == (il_byte, xl_byte)
    assert layout.sort == sort
    assert layout.shape == cube.shape
    assert (layout.il_min, layout.il_step, layout.xl_min, layout.xl_step) == (100, 2, 1000, 1)
    exact, cellmap = scan_layout_exact(reader, layout)
    assert exact.regular and cellmap is None and not exact.missing

    decoded = reader.read(0, reader.n_traces)
    with segyio.open(str(path), ignore_geometry=True) as f:
        reference = segyio.tools.collect(f.trace[:])
    np.testing.assert_array_equal(decoded, reference)
    reader.close()

    directory = tmp_path / "cache"
    manifest = build_cache(path, directory)
    assert read_manifest(directory) == manifest
    _assert_orders(directory, manifest, _expected_u8(cube, manifest))


def test_missing_traces_are_filled(tmp_path):
    cube = _cube(2)
    missing = {(3, 4), (3, 5), (7, 11), (10, 2)}
    path = write_segy(tmp_path / "holes.sgy", cube, drop=missing)
    directory = tmp_path / "cache"
    manifest = build_cache(path, directory)
    layout = manifest["layout"]
    assert tuple(manifest["shape"]) == cube.shape
    assert layout["missing"] == len(missing)
    assert manifest["has_cellmap"]
    _assert_orders(directory, manifest, _expected_u8(cube, manifest, missing))


def test_cache_matches_in_memory_path(tmp_path):
    """The cached slices are what the old in-memory path showed (within 1 level)."""
    cube = _cube(3)
    path = write_segy(tmp_path / "match.sgy", cube)
    data, geometry = load_volume(path)
    data_u8 = normalize_to_uint8(data)

    state = BuildState(name=path.name, path=path, directory=tmp_path / "cache")
    manager = CacheManager(root=tmp_path, use_process=False)
    manager._build(state)
    assert state.ready, state.error
    volume = CachedVolume(path.name, path, state)
    assert volume.shape == data.shape
    for axis, count in (("inline", N_IL), ("crossline", N_XL), ("time", N_S)):
        for index in (0, count // 2, count - 1):
            ours = volume.rgb(axis, index).astype(np.int16)
            theirs = slice_rgb(data_u8, geometry, axis, index).astype(np.int16)
            assert ours.shape == theirs.shape
            assert np.abs(ours - theirs).max() <= 1, (axis, index)
    volume.close()


# ---- direct reads while the cache builds -------------------------------------


@pytest.mark.parametrize("sort", ["inline", "crossline"])
def test_direct_volume_serves_storage_axis(tmp_path, sort):
    cube = _cube(4)
    path = write_segy(tmp_path / f"direct_{sort}.sgy", cube, sort=sort)
    directory = tmp_path / "cache"
    manager = CacheManager(root=tmp_path, use_process=False)

    # Replay every event except "done": the state a volume sees mid-build.
    events = []
    build_cache(path, directory, events.append)
    building = BuildState(name=path.name, path=path, directory=directory)
    for event in events:
        if event["type"] != "done":
            manager._apply(building, event)
    assert not building.ready

    listing = inspect_any_for_listing(path)
    direct = DirectVolume(path.name, path, building, listing)
    storage = "inline" if sort == "inline" else "crossline"
    other = "crossline" if sort == "inline" else "inline"
    ready = direct.axes_ready()
    assert ready[storage] and not ready[other] and not ready["time"]
    with pytest.raises(AxisNotReady):
        direct.rgb(other, 0)

    finished = BuildState(name=path.name, path=path, directory=directory)
    manager._mark_ready(finished, read_manifest(directory))
    cached = CachedVolume(path.name, path, finished)
    for index in range(direct.axis_count(storage)):
        np.testing.assert_array_equal(direct.rgb(storage, index), cached.rgb(storage, index))
    direct.close()
    cached.close()


def test_direct_volume_before_layout_is_known(tmp_path):
    path = write_segy(tmp_path / "early.sgy", _cube(5))
    state = BuildState(name=path.name, path=path, directory=tmp_path / "cache")
    direct = DirectVolume(path.name, path, state, inspect_any_for_listing(path))
    assert direct.shape == (N_IL, N_XL, N_S)
    assert not any(direct.axes_ready().values())
    with pytest.raises(AxisNotReady):
        direct.rgb("inline", 0)


# ---- 2D lines and NumPy ------------------------------------------------------


def test_wide_2d_line_is_paged(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_2D_PAGE_TRACES", 64)
    monkeypatch.setattr(config, "PAGE_OVERLAP_TRACES", 16)
    section = _cube(6, shape=(200, 30))[:, :]
    path = write_segy_2d(tmp_path / "line.sgy", section)

    pages = make_pages(200)
    assert (pages.count, pages.width, pages.step) == (4, 64, 48)
    assert pages.bounds(pages.count - 1) == (136, 200)

    directory = tmp_path / "cache"
    manifest = build_cache(path, directory)
    assert manifest["kind"] == "2d"
    assert manifest["orders"] == ["section"]
    state = BuildState(name=path.name, path=path, directory=directory)
    CacheManager(root=tmp_path, use_process=False)._mark_ready(state, manifest)
    volume = CachedVolume(path.name, path, state)
    assert volume.axis_count("inline") == pages.count
    assert volume.frame_shape("inline") == (30, 64)

    expected = normalize_block(section.copy(), manifest["lo"], manifest["hi"]).T
    for page in range(pages.count):
        a, b = pages.bounds(page)
        np.testing.assert_array_equal(volume.rgb("inline", page), to_rgb(expected[:, a:b]))


def test_large_npy_cache(tmp_path):
    cube = _cube(7, shape=(9, 14, 25))
    cube[2, 3, 4] = np.nan
    path = tmp_path / "volume.npy"
    np.save(path, cube)
    directory = tmp_path / "cache"
    manifest = build_cache(path, directory)
    clean = np.nan_to_num(cube, nan=0.0)
    _assert_orders(directory, manifest, _expected_u8(clean, manifest))


# ---- cache manager -----------------------------------------------------------


def _wait_ready(state: BuildState, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while not state.ready and state.error is None and time.time() < deadline:
        time.sleep(0.05)
    assert state.ready, state.error or f"still {state.stage}"


def test_manager_builds_hits_and_invalidates(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = write_segy(data_dir / "managed.sgy", _cube(8))
    manager = CacheManager(root=tmp_path / "cache", use_process=False)
    try:
        state = manager.ensure(path.name, path, priority=True)
        _wait_ready(state)
        first_dir = state.directory
        assert (first_dir / "manifest.json").exists()

        # A fresh manager finds the cache on disk without rebuilding.
        again = CacheManager(root=tmp_path / "cache", use_process=False)
        hit = again.ensure(path.name, path)
        assert hit.ready and hit.directory == first_dir
        again.shutdown()

        # Rewriting the source invalidates the cache and removes the stale one.
        time.sleep(0.05)
        write_segy(path, _cube(9), il_step=1)
        assert cache_dir_for(path, manager.root) != first_dir
        rebuilt = manager.ensure(path.name, path)
        _wait_ready(rebuilt)
        assert rebuilt.directory != first_dir
        assert not first_dir.exists()
        assert rebuilt.to_json()["percent"] == 100.0
    finally:
        manager.shutdown()


def test_partial_cache_is_not_trusted(tmp_path):
    path = write_segy(tmp_path / "partial.sgy", _cube(10))
    directory = cache_dir_for(path, tmp_path / "cache")
    build_cache(path, directory)
    (directory / "manifest.json").unlink()
    assert read_manifest(directory) is None
    manifest = build_cache(path, directory)
    (directory / "time.u8").write_bytes(b"\0" * 10)
    assert read_manifest(directory) is None
    assert manifest["complete"]


# ---- server routing and windowed propagation ---------------------------------


def test_server_routes_large_files_to_the_cache(tmp_path, monkeypatch):
    from server import main

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = write_segy(data_dir / "big.sgy", _cube(11))
    manager = CacheManager(root=tmp_path / "cache", use_process=False)
    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "_cache_manager", manager)
    monkeypatch.setattr(main, "_routes", {})
    monkeypatch.setattr(main, "_large_volumes", {})
    monkeypatch.setattr(config, "IN_MEMORY_BUDGET_GB", "0.0000001")

    def forbid_load(*_args, **_kwargs):
        raise AssertionError("large files must not be loaded into memory")

    monkeypatch.setattr(main, "load_any", forbid_load)
    try:
        meta = main.file_meta(path.name)
        assert meta["large"] and tuple(meta["shape"]) == (N_IL, N_XL, N_S)
        state = manager.state(path.name)
        _wait_ready(state)
        status = main.file_status(path.name)
        assert status["status"]["ready"]
        assert all(status["axes_ready"].values())
        for axis in ("inline", "crossline", "time"):
            response = main.get_slice(path.name, axis, 1)
            assert response.status_code == 200
            assert response.body[:2] == b"\xff\xd8"
        listing = {f["name"]: f for f in main.list_files()}
        assert listing[path.name]["large"]
    finally:
        for volume in main._large_volumes.values():
            getattr(volume, "close", lambda: None)()
        manager.shutdown()


def _fake_volume(n: int, large: bool, frame=(100, 200)):
    return SimpleNamespace(
        name="fake",
        large=large,
        axis_count=lambda _axis: n,
        frame_shape=lambda _axis: frame,
    )


def test_propagation_window_in_memory_is_full_axis():
    from server import main

    assert main._propagation_window(_fake_volume(500, False), "inline", 250, [250], 1, None)[:2] == (0, 500)


def test_propagation_window_requested_and_clamped():
    from server import main

    volume = _fake_volume(100, False)
    assert main._propagation_window(volume, "inline", 50, [50], 1, 5)[:2] == (45, 56)
    assert main._propagation_window(volume, "inline", 2, [2], 1, 10)[:2] == (0, 13)
    assert main._propagation_window(volume, "inline", 98, [98], 1, 10)[:2] == (88, 100)
    # Every prompted slice stays inside the window.
    assert main._propagation_window(volume, "inline", 50, [30, 50, 71], 2, 5)[:2] == (30, 72)


def test_propagation_window_auto_sizes_large_volumes(monkeypatch):
    from server import main

    frame = (1000, 2000)
    per_frame = frame[0] * frame[1] * 4 + 3 * config.PROPAGATION_MODEL_SIDE**2 * 4
    budget_frames = 40
    total = int(budget_frames * per_frame / config.PROPAGATION_RAM_FRACTION)
    monkeypatch.setattr(main.sysinfo, "memory_status", lambda: (total, total))
    start, stop, reason = main._propagation_window(_fake_volume(2000, True, frame), "time", 1000, [1000], 1, None)
    assert start < 1000 < stop
    assert stop - start <= budget_frames + 1
    assert stop - start >= config.PROPAGATION_MIN_FRAMES
    assert "auto" in reason
    # A large volume that fits the budget is still tracked end to end.
    assert main._propagation_window(_fake_volume(20, True, frame), "time", 10, [10], 1, None)[:2] == (0, 20)


def test_propagation_local_frame_mapping():
    from server import main

    prop = main.Propagation(file="f", axis="inline", object_ids=[0], live=None, start=40, stop=60)
    assert prop.local_frame(40) == 0
    assert prop.local_frame(59) == 19
    assert prop.local_frame(39) is None
    assert prop.local_frame(60) is None
