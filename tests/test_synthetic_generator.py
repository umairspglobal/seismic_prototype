"""Synthetic seismic generator: config, labels, geology, dataset layout, CLI, previews and GUI."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import fields

import numpy as np
import pytest

from synthetic_generator import labels as lab
from synthetic_generator.block import Product, generate_block, products
from synthetic_generator.config import ARRAY_KINDS, PRESETS, GeneratorConfig, LabelConfig, preset
from synthetic_generator.dataset import (block_id_list, generate_dataset, next_start_id, plan_blocks,
                                         read_block_from_disk)
from synthetic_generator.geology import Fault, Package, Stratigraphy, build_slice
from synthetic_generator.params import all_params
from synthetic_generator.seismic import gaussianize


def small_config(**changes) -> GeneratorConfig:
    cfg = GeneratorConfig()
    cfg.grid.height = 64
    cfg.grid.width = 64
    cfg.grid.n_slices = 3
    cfg.grid.n_replicas = 2
    cfg.grid.scales = [6, 12]
    cfg.structure.fold_width = (6.0, 20.0)
    cfg.structure.fold_amplitude = (1.0, 6.0)
    cfg.faults.throw = (2.0, 6.0)
    cfg.output.workers = 1
    for path, value in changes.items():
        cfg.set(path.replace("__", "."), value)
    return cfg


# --- config -----------------------------------------------------------------------------


def test_default_config_and_presets_are_valid():
    assert GeneratorConfig().validate() == []
    for name in PRESETS:
        assert preset(name).validate() == [], name


def test_config_round_trip_and_coercion(tmp_path):
    cfg = small_config()
    cfg.seismic.centers = [0.05, 0.1]
    path = tmp_path / "cfg.json"
    cfg.save(path)
    loaded = GeneratorConfig.load(path)
    assert loaded.to_dict() == cfg.to_dict()
    loaded.set("faults.count", ["2", "5"])
    loaded.set("grid.scales", "8, 16")
    loaded.set("output.overwrite", "true")
    assert loaded.faults.count == (2, 5)
    assert loaded.grid.scales == [8, 16]
    assert loaded.output.overwrite is True


def test_validation_reports_bad_values():
    cfg = GeneratorConfig()
    cfg.faults.count = (5, 2)
    cfg.grid.scales = [300]
    cfg.seismic.wavelet = "nope"
    cfg.faults.dip_spread = -1.0
    cfg.grid.width = 200
    errors = " ".join(cfg.validate())
    assert "Fault count" in errors and "Scale S=300" in errors and "Wavelet" in errors
    assert "Dip spread cannot be negative" in errors and "square" in errors


_FUZZ_CAPS = {
    "structure.t0": 200, "structure.fold_count": 6, "structure.fold_amplitude": 60, "structure.fold_width": 120,
    "structure.unconformity_relief": 40, "structure.unconformity_hiatus": 120, "structure.slice_drift": 6,
    "faults.count": 6, "faults.segments": 4, "faults.throw": 40, "faults.tip_taper": 40, "faults.drag_length": 40,
    "faults.step": 6, "faults.min_spacing": 30, "faults.kink_jitter": 10, "faults.listric": 20,
    "faults.step_jitter": 2, "faults.dip_spread": 20, "seismic.layer_thickness": 12, "seismic.lateral_length": 300,
    "seismic.supersample": 4, "seismic.clip": 6, "seismic.marker_strength": 5, "realism.gain_length": 120,
    "realism.chaos_count": 3, "realism.chaos_size": 120, "realism.noise_snr": 50,
    "realism.fault_dimming_width": 8, "realism.noise_lateral": 4, "realism.lateral_smoothing": 4,
    "realism.trace_jitter": 2, "labels.dip_smoothing": 4, "labels.corridor_pad": 8, "labels.partner_pad_left": 8,
    "labels.partner_pad_right": 8,
}


def test_random_valid_configs_always_generate():
    """Anything validate() accepts must generate (fuzzed over every GUI control)."""
    rng = np.random.default_rng(1234)
    valid = 0
    for trial in range(40):
        cfg = small_config()
        for p in all_params():
            if p.path.startswith(("output.", "grid.")):
                continue
            lo = 0.0 if p.lo is None else p.lo
            hi = min(1.0 if p.hi is None else p.hi, _FUZZ_CAPS.get(p.path, float("inf")))
            if p.kind == "prob":
                cfg.set(p.path, float(rng.random()))
            elif p.kind == "bool":
                cfg.set(p.path, bool(rng.random() < 0.5))
            elif p.kind == "choice":
                cfg.set(p.path, str(rng.choice(p.choices)))
            elif p.kind == "range_int":
                cfg.set(p.path, tuple(sorted(int(v) for v in rng.integers(int(lo), int(hi) + 1, 2))))
            elif p.kind == "range_float":
                cfg.set(p.path, tuple(sorted(float(v) for v in rng.uniform(lo, hi, 2))))
            elif p.kind == "int":
                cfg.set(p.path, int(rng.integers(int(lo), int(hi) + 1)))
            elif p.kind == "float":
                cfg.set(p.path, float(rng.uniform(lo, hi)))
            elif p.path == "seismic.centers":
                cfg.set(p.path, [float(v) for v in rng.uniform(0.02, 0.4, 3)])
        cfg.faults.dip_median = float(rng.uniform(*cfg.faults.dip_limits))
        if cfg.validate():
            continue
        valid += 1
        block = generate_block(cfg, trial)
        assert np.isfinite(block.seis).all() and np.isfinite(block.tau).all(), trial
    assert valid >= 20


def test_every_gui_param_maps_to_a_config_field():
    cfg = GeneratorConfig()
    names = {f"{s.name}.{f.name}" for s in fields(cfg) for f in fields(getattr(cfg, s.name))}
    for p in all_params():
        assert p.path in names, p.path
        value = cfg.get(p.path)
        if p.kind.startswith("range"):
            assert isinstance(value, tuple) and len(value) == 2, p.path
        elif p.kind == "bool":
            assert isinstance(value, bool), p.path
        elif p.kind in ("list", "arrays"):
            assert isinstance(value, list), p.path


# --- labels -----------------------------------------------------------------------------


def test_window_any_matches_brute_force():
    rng = np.random.default_rng(0)
    mask = rng.random((5, 30)) < 0.1
    for lo, hi in ((-2, 2), (-5, 2), (0, 7), (3, 3)):
        got = lab.window_any(mask, lo, hi)
        want = np.array([[mask[y, max(0, x + lo):max(0, min(30, x + hi + 1))].any() for x in range(30)]
                         for y in range(5)])
        assert np.array_equal(got, want), (lo, hi)


def test_dip_is_the_tau_difference():
    y, x = np.mgrid[0:20, 0:40].astype(float)
    tau = 10 + y + 0.25 * x
    d = lab.dip(tau, 8, smoothing=0.0)
    assert np.allclose(d, -2.0)
    assert np.allclose(lab.dip(tau, 8, smoothing=1.0)[:, :20], -2.0)
    assert np.allclose(lab.dip(tau, 8, smoothing=0.0, fill="zero")[:, -8:], 0.0)


def _vertical_fault(H=10, W=40, left=20, width=4):
    mask = np.zeros((H, W), bool)
    mask[:, left:left + width] = True
    X = np.tile(np.arange(W, dtype=float), (H, 1))
    center = left + (width - 1) / 2
    return mask, [X - center], [np.ones((H, W), bool)]


def test_straddle_exact_predicate():
    mask, offsets, inside = _vertical_fault(left=20, width=4)
    st = lab.straddle(mask, offsets, inside, S=6)
    xs = np.nonzero(st[0])[0].tolist()
    # x off the stroke (< 20), x + 6 off the stroke (> 23), trace between them.
    assert xs == [18, 19]
    inside[0][:3] = False
    assert not lab.straddle(mask, offsets, inside, S=6)[:3].any()


def test_weight_right_edge_corridors_and_the_s12_notch():
    mask, _, _ = _vertical_fault(W=80, left=40, width=4)
    tau = np.tile(np.arange(10, dtype=float)[:, None], (1, 80)) + 100
    w = lab.weight(mask, tau, 12, LabelConfig())
    row = w[5]
    assert not row[80 - 12 - 1:].any()
    assert not row[38:46].any()
    # Partner corridor for x + 12 in [38, 48] -> x in [26, 36]; one-pixel notch at 37 (3 columns before 40).
    assert not row[26:37].any()
    assert row[37] == 1.0 and row[25] == 1.0


def test_weight_zeroes_pairs_across_packages():
    mask = np.zeros((6, 40), bool)
    tau = np.tile(np.arange(6, dtype=float)[:, None], (1, 40))
    package = np.zeros((6, 40), np.int8)
    package[:, 25:] = 1
    w = lab.weight(mask, tau, 5, LabelConfig(), package)
    assert not w[:, 20:25].any() and w[:, :20].all()


def test_halo_peaks_at_one():
    mask, _, _ = _vertical_fault(H=40, W=60, left=30, width=3)
    h = lab.halo(mask, 12, LabelConfig())
    assert h.max() == pytest.approx(1.0)
    assert h[:, 0].max() == 0.0 and h.min() >= 0.0
    assert not lab.halo(np.zeros((8, 8), bool), 4, LabelConfig()).any()


# --- geology / seismic ---------------------------------------------------------------------


def test_throw_sits_on_the_footwall_edge_of_the_stroke():
    cfg = small_config()
    H, W = cfg.grid.height, cfg.grid.width
    strat = Stratigraphy(packages=[Package(folds=[], tilt=0.0, depth_scale=1.0, thickening=0.0)], surfaces=[], H=H, W=W)
    ys = np.array([-20.0, H + 20.0])
    xs = np.array([40.0, 40.0 - (ys[1] - ys[0]) / np.tan(np.radians(75))])
    fault = Fault(xs=xs, ys=ys, dip_dir=-1, radius=1.7, throw=6.0, profile="constant", peak=0.5, taper=0.0,
                  offsets=np.zeros(3), throw_rate=0.0, drag=0.0, drag_strength=0.0, order=0, dip=75.0, through=True)
    geom = build_slice(strat, [fault], 1, cfg, pad=0)
    y = H // 2
    stroke = np.nonzero(geom.mask[y])[0]
    xl, xr = stroke[0], stroke[-1]
    assert 3 <= stroke.size <= 4
    jumps = np.diff(geom.tau[y])
    assert jumps[xr] == pytest.approx(6.0)
    assert np.allclose(jumps[:xr], 0.0) and np.allclose(jumps[xr + 1:], 0.0)
    assert np.allclose(geom.tau[y, xl - 1:xr + 1], geom.tau[y, 0])


def test_block_shapes_determinism_and_label_ranges():
    cfg = small_config()
    a = generate_block(cfg, 3)
    b = generate_block(cfg, 3)
    c = generate_block(cfg, 4)
    assert a.seis.shape == (2, 3, 64, 64) and a.tau.shape == (3, 64, 64) and a.fault.shape == (3, 64, 64)
    assert np.array_equal(a.seis, b.seis) and np.array_equal(a.tau, b.tau)
    assert not np.array_equal(a.tau, c.tau)
    assert set(np.unique(a.fault)) <= {0.0, 1.0}
    for s in range(3):
        for S in cfg.grid.scales:
            assert set(np.unique(a.labels[s][("straddle", S)])) <= {0.0, 1.0}
            assert set(np.unique(a.labels[s][("weight", S)])) <= {0.0, 1.0}
            halo = a.labels[s][("halo", S)]
            assert halo.min() >= 0 and halo.max() <= 1
    for arr in (a.seis, a.tau):
        assert np.isfinite(arr).all()


def test_geology_does_not_depend_on_seismic_settings():
    a = generate_block(small_config(), 1)
    b = generate_block(small_config(realism__noise_snr=(1.0, 1.0), seismic__wavelet="ormsby"), 1)
    assert np.array_equal(a.tau, b.tau) and np.array_equal(a.fault, b.fault)
    assert not np.array_equal(a.seis, b.seis)


def test_spec_mode_keeps_reflections_and_steps_faults_right():
    cfg = preset("Spec: fixed reflections, 2-trace step")
    cfg.faults.count = (1, 1)
    cfg.faults.p_through = 1.0
    block = generate_block(cfg, 7)
    tau = block.tau
    far = np.ones(tau.shape[1:], bool)
    for m in block.fault:
        far &= ~lab.window_any(m > 0.5, -40, 40)
    assert np.allclose(tau[0][far], tau[-1][far])
    cols = [np.nonzero(block.fault[s][112])[0].mean() for s in range(cfg.grid.n_slices)]
    assert np.allclose(np.diff(cols), 2.0, atol=0.6)


def test_gaussianize_and_clip():
    x = np.random.default_rng(1).standard_exponential(5000) ** 3
    g = gaussianize(x)
    assert abs(g.mean()) < 1e-3 and abs(g.std() - 1) < 0.02
    assert np.array_equal(np.argsort(g, kind="stable"), np.argsort(x, kind="stable"))
    cfg = small_config(seismic__clip_prob=1.0, seismic__clip=(2.5, 2.5))
    assert np.abs(generate_block(cfg, 0).seis).max() == pytest.approx(2.5)


# --- dataset ------------------------------------------------------------------------------------


def test_dataset_layout_resume_and_finetuner_compatibility(tmp_path):
    cfg = small_config()
    cfg.output.num_blocks = 2
    events = []
    summary = generate_dataset(cfg, tmp_path, events.append)
    assert summary["written"] == 2 and summary["next_id"] == 12
    folders = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert folders == sorted(p.folder("75") for p in products(cfg))
    assert {"Seis75", "Fault75", "TauAtT75", "dipXscale6_75", "wtXStraddleScale12_75", "wtXHaloScale6_75"} <= set(folders)
    assert len(list((tmp_path / "Seis75").glob("*.dat"))) == 12
    assert (tmp_path / "Seis75" / "0.dat").stat().st_size == 64 * 64 * 8

    assert block_id_list(cfg, 0, 1) == [(6, 0, 0), (7, 0, 1), (8, 0, 2), (9, 1, 0), (10, 1, 1), (11, 1, 2)]
    data = read_block_from_disk(tmp_path, cfg, 0, 1)
    for prod, arr in data.items():
        if prod.kind != "seis":
            assert np.array_equal(arr[0], arr[1]), prod
    assert not np.array_equal(data[Product("seis")][0], data[Product("seis")][1])
    block = generate_block(cfg, 1)
    assert np.array_equal(data[Product("fault")][0], block.fault)

    cfg.output.start_id = 0
    again = generate_dataset(cfg, tmp_path)
    assert again["written"] == 0 and again["skipped"] == 2
    cfg.output.start_id = -1
    assert next_start_id(tmp_path, cfg) == 12
    cfg.output.num_blocks = 1
    appended = generate_dataset(cfg, tmp_path)
    assert appended["start_id"] == 12 and appended["written"] == 1

    rows = list(csv.DictReader(open(tmp_path / "samples.csv", encoding="utf-8")))
    assert len(rows) == 18 and rows[4]["replica"] == "1" and rows[4]["slice"] == "1" and rows[7]["block"] == "1"
    run = json.loads((tmp_path / "generator_config.json").read_text(encoding="utf-8"))["run"]
    assert run["start_id"] == 12 and run["folders"]["seis"] == "Seis75"

    from finetuning.quick_data import list_pairs, load_sections, read_section

    pairs = list_pairs(tmp_path / "Seis75", tmp_path / "Fault75")
    assert pairs == [str(i) for i in range(18)]
    assert read_section(tmp_path / "Fault75" / "3.dat").shape == (64, 64)
    sections = load_sections(tmp_path / "Seis75", tmp_path / "Fault75", pairs[:4], "binary")
    assert sections.masks.any()
    layers = load_sections(tmp_path / "Seis75", tmp_path / "TauAtT75", pairs[:2], "rgt_layers")
    assert 0.2 < layers.positive_fraction < 0.8


def test_subset_of_arrays_and_float32(tmp_path):
    cfg = small_config(output__dtype="float32")
    cfg.output.arrays = ["seis", "fault"]
    cfg.output.num_blocks = 1
    generate_dataset(cfg, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == ["Fault75", "Seis75"]
    assert (tmp_path / "Fault75" / "0.dat").stat().st_size == 64 * 64 * 4


def test_plan_counts_only_missing_files_and_requires_a_folder(tmp_path):
    cfg = small_config()
    cfg.output.num_blocks = 2
    cfg.output.arrays = ["seis", "fault"]
    size = 64 * 64 * 8
    pending, skipped, needed = plan_blocks(tmp_path, cfg, 0)
    assert pending == [0, 1] and skipped == 0 and needed == 2 * 2 * 6 * size
    generate_dataset(cfg, tmp_path)
    (tmp_path / "Fault75" / "8.dat").unlink()
    pending, skipped, needed = plan_blocks(tmp_path, cfg, 0)
    assert pending == [1] and skipped == 1 and needed == size
    cfg.output.overwrite = True
    assert plan_blocks(tmp_path, cfg, 0)[:2] == ([0, 1], 0)
    cfg.output.out_dir = "  "
    with pytest.raises(ValueError, match="No output folder"):
        generate_dataset(cfg)


def test_cli_generate_and_inspect(tmp_path, capsys):
    from synthetic_generator.cli import main

    cfg = small_config()
    cfg_path = tmp_path / "cfg.json"
    cfg.save(cfg_path)
    out = tmp_path / "data"
    assert main(["generate", "--config", str(cfg_path), "--out", str(out), "--blocks", "1", "--events"]) == 0
    lines = [json.loads(l[2:]) for l in capsys.readouterr().out.splitlines() if l.startswith("@@")]
    assert [e["event"] for e in lines][0] == "start" and lines[-1]["event"] == "done"
    assert main(["inspect", str(out), "--sample", "6"]) == 0
    assert "Quick Fine-Tuner pairs (Seis/Fault): 6" in capsys.readouterr().out


# --- previews / GUI ---------------------------------------------------------------------------------


@pytest.mark.parametrize("view", ["overview", "arrays", "slices", "replicas", "labels", "sam3"])
def test_every_preview_view_renders(view):
    from synthetic_generator.preview import fit_tile, render_view

    cfg = small_config()
    block = generate_block(cfg, 2)
    image, infos = render_view(block, cfg, view, replica=1, slice_index=2, scale=12, tile=96)
    assert image.width > 96 and image.height > 96
    assert any(i.array is not None for i in infos)
    assert fit_tile(view, cfg, 1200, 700) >= 64


def _tk_root():
    """A Tk root, retrying the occasional transient Tcl start-up failure seen on Windows."""
    tk = pytest.importorskip("tkinter")
    error = None
    for _ in range(3):
        try:
            return tk.Tk()
        except tk.TclError as exc:
            error = exc
            time.sleep(0.3)
    pytest.skip(f"Tk display is not available: {error}")


def test_gui_builds_and_reads_settings(monkeypatch, tmp_path):
    root = _tk_root()
    import synthetic_generator.gui as gui_mod

    monkeypatch.setattr(gui_mod, "SETTINGS_FILE", tmp_path / "gui.json")
    try:
        root.withdraw()
        gui = gui_mod.GeneratorGUI(root)
        root.update()
        assert root.cget("bg") == gui_mod.APP_BG
        cfg, errors = gui.read_config()
        assert errors == [] and cfg.validate() == []
        gui.vars["faults.stroke_radius"][1].set("2.5")
        gui.array_vars["halo"].set(False)
        cfg, errors = gui.read_config()
        assert cfg.faults.stroke_radius[1] == 2.5 and "halo" not in cfg.output.arrays
        gui.vars["faults.count"][0].set("abc")
        assert gui.read_config()[1]
        gui.preset_var.set("Spec: fixed reflections, 2-trace step")
        gui._apply_preset()
        cfg, errors = gui.read_config()
        assert errors == [] and cfg.faults.step_direction == "right" and cfg.structure.slice_drift == (0.0, 0.0)

        gui.slice_var.set(99)
        gui.replica_var.set(-4)
        gui.block_var.set(-2)
        gui.request_preview()
        job = gui._pending_job
        assert (job.slice_index, job.replica, job.block_index) == (cfg.grid.n_slices - 1, 0, 0)

        launched = []
        monkeypatch.setattr(gui_mod.subprocess, "Popen", lambda *a, **k: launched.append(a))
        monkeypatch.setattr(gui_mod.messagebox, "showinfo", lambda *a, **k: None)
        gui.proc = object()
        gui.start()
        gui.resume()
        assert launched == []
    finally:
        if "gui" in locals():
            gui.proc = None
            gui.shutdown()
        root.destroy()


def test_send_to_finetuner_writes_its_settings(monkeypatch, tmp_path):
    root = _tk_root()
    import finetuning.finetune_gui as ft
    import synthetic_generator.gui as gui_mod

    monkeypatch.setattr(gui_mod, "SETTINGS_FILE", tmp_path / "gui.json")
    monkeypatch.setattr(ft, "SETTINGS_FILE", tmp_path / "ft.json")
    launched = []
    monkeypatch.setattr(gui_mod.subprocess, "Popen", lambda *a, **k: launched.append(a[0]))
    data = tmp_path / "data"
    for name in ("Seis75", "Fault75", "TauAtT75"):
        (data / name).mkdir(parents=True)
    try:
        root.withdraw()
        gui = gui_mod.GeneratorGUI(root)
        gui.vars["output.out_dir"].set(str(data))
        gui.send_to_finetuner("faults")
        saved = json.loads((tmp_path / "ft.json").read_text(encoding="utf-8"))
        assert saved["image_dir"] == str(data / "Seis75") and saved["label_dir"] == str(data / "Fault75")
        assert ft.PRESETS[saved["preset"]]["label"] == "Fault75" and saved["prompt"] == "fault"
        gui.send_to_finetuner("facies")
        saved = json.loads((tmp_path / "ft.json").read_text(encoding="utf-8"))
        assert saved["label_dir"] == str(data / "TauAtT75") and ft.LABEL_MODES[saved["label_mode"]] == "rgt_layers"
        assert launched and launched[0][1:] == ["-m", "finetuning.finetune_gui"]
    finally:
        if "gui" in locals():
            gui.shutdown()
        root.destroy()
