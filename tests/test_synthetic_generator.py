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
from synthetic_generator.dataset import (block_cpu_weight, block_id_list, estimate_run_seconds, generate_dataset,
                                         measure_write_bps, next_start_id, plan_blocks, read_block_from_disk)
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


def test_default_and_csd_presets():
    from synthetic_generator.config import DEFAULT_PRESET, default_config

    assert DEFAULT_PRESET == "Noisy field-like data" and "Reference match (training_10_7_26)" not in PRESETS
    assert default_config().realism.noise_snr == (1.2, 4.0) and not default_config().csd.enabled
    cfg = preset("CSD: horizon tracks + dip prompts")
    assert cfg.csd.enabled and cfg.output.arrays == ["seis", "dip", "weight"] and cfg.realism.noise_snr == (1.2, 4.0)
    cfg.csd.count = (2, 2)
    back = preset("Complex structure", cfg)
    assert not back.csd.enabled and back.csd.count == (2, 2)


def test_finetune_presets_match_generator_folders():
    from finetuning.presets import PRESETS as FT
    from finetuning.presets import folders_for
    from synthetic_generator.block import products

    suffix = GeneratorConfig().output.suffix
    live = {v["task"]: (k, v) for k, v in FT.items() if v}
    assert list(live) == ["faults", "facies", "csd"]
    assert list(FT)[-1] == "Custom" and FT["Custom"] is None

    expected = {
        "faults": ("fault", "binary", "fault"),
        "facies": ("tau", "rgt_layers", "seismic facies"),
        "csd": ("csd_mask", "binary", "horizon"),
    }
    for task, (kind, mode, prompt) in expected.items():
        key, spec = live[task]
        image, label = Product("seis").folder(suffix), Product(kind).folder(suffix)
        assert spec["image"] == image and spec["label"] == label
        assert spec["mode"] == mode and spec["prompt"] == prompt
        assert image in key and label in key
        assert folders_for(spec, "80") == (Product("seis").folder("80"), Product(kind).folder("80"))

    noisy = {p.folder(suffix) for p in products(preset("Noisy field-like data"))}
    assert live["faults"][1]["image"] in noisy and live["faults"][1]["label"] in noisy
    assert live["facies"][1]["label"] in noisy
    faults_only = {p.folder(suffix) for p in products(preset("Faults only (Seis + Fault folders)"))}
    assert faults_only == {live["faults"][1]["image"], live["faults"][1]["label"]}
    csd_names = {p.folder(suffix) for p in products(preset("CSD: horizon tracks + dip prompts"))}
    assert live["csd"][1]["image"] in csd_names and live["csd"][1]["label"] in csd_names
    assert live["faults"][1]["label"] not in csd_names and live["facies"][1]["label"] not in csd_names


def test_dataset_suffix_follows_the_generator(tmp_path):
    from finetuning.presets import DEFAULT_SUFFIX, dataset_suffix

    assert dataset_suffix(tmp_path / "missing") == DEFAULT_SUFFIX
    (tmp_path / "Seis80").mkdir()
    assert dataset_suffix(tmp_path) == "80"
    (tmp_path / "Seis75").mkdir()
    assert dataset_suffix(tmp_path) == DEFAULT_SUFFIX
    (tmp_path / "generator_config.json").write_text(json.dumps({"output": {"suffix": "80"}}), encoding="utf-8")
    assert dataset_suffix(tmp_path) == "80"


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
    "labels.partner_pad_right": 8, "csd.count": 10, "csd.min_separation": 30, "csd.gap_length": 30,
    "csd.thickness": 4, "csd.pick_jitter": 5, "csd.jitter_length": 40, "csd.prompt_noise": 5,
    "csd.prompt_background": 5, "csd.distance_clip": 60,
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


# --- coarse-scale dips (CSD) ---------------------------------------------------------------------


def _faulted_section(throw=6.0):
    cfg = small_config()
    H, W = cfg.grid.height, cfg.grid.width
    strat = Stratigraphy(packages=[Package(folds=[], tilt=0.0, depth_scale=1.0, thickening=0.0)], surfaces=[], H=H, W=W)
    ys = np.array([-20.0, H + 20.0])
    xs = np.array([40.0, 40.0 - (ys[1] - ys[0]) / np.tan(np.radians(75))])
    fault = Fault(xs=xs, ys=ys, dip_dir=-1, radius=1.7, throw=throw, profile="constant", peak=0.5, taper=0.0,
                  offsets=np.zeros(3), throw_rate=0.0, drag=0.0, drag_strength=0.0, order=0, dip=75.0, through=True)
    return cfg, build_slice(strat, [fault], 1, cfg, pad=0)


def _plan(**changes):
    from synthetic_generator.csd import HorizonPlan

    values = dict(depth=0.4, window=(0.0, 64.0), seed_x=5.0, gaps=[], thickness=1, connector="line", seed=1,
                  in_prompt=True)
    values.update(changes)
    return HorizonPlan(**values)


@pytest.mark.parametrize("connector", ["line", "smooth", "step"])
def test_csd_horizon_steps_down_at_a_fault_and_continues(connector):
    from synthetic_generator.csd import section_csd

    cfg, geom = _faulted_section(throw=6.0)
    cfg.csd.connect_prob, cfg.csd.p_stop_at_fault = 1.0, 0.0
    labels = lab.section_labels(geom.tau, geom.mask, geom.offsets, geom.inside, cfg.grid.scales, cfg.labels)
    out, drawn = section_csd(geom.tau, geom.mask.astype(float), [_plan(connector=connector)], labels,
                             cfg.grid.scales, cfg.csd, np.random.default_rng(0))
    m = out[("csd_mask", None)] > 0.5
    assert drawn == 1 and m.any(axis=0).all()
    rows = [np.nonzero(m[:, x])[0].mean() for x in (2, 60)]
    assert rows[0] - rows[1] == pytest.approx(6.0, abs=1.0)
    assert ndi_label(m)[1] == 1

    cfg.csd.connect_prob = 0.0
    out, _ = section_csd(geom.tau, geom.mask.astype(float), [_plan()], labels, cfg.grid.scales, cfg.csd,
                         np.random.default_rng(0))
    assert ndi_label(out[("csd_mask", None)] > 0.5)[1] == 2
    cfg.csd.connect_prob, cfg.csd.p_stop_at_fault = 1.0, 1.0
    out, _ = section_csd(geom.tau, geom.mask.astype(float), [_plan(seed_x=5.0)], labels, cfg.grid.scales,
                         cfg.csd, np.random.default_rng(0))
    assert not (out[("csd_mask", None)][:, 45:] > 0.5).any()


def ndi_label(mask):
    from scipy import ndimage as ndi

    return ndi.label(mask, structure=np.ones((3, 3)))


def test_csd_block_arrays_prompt_and_distance():
    cfg = small_config(csd__enabled=True, csd__count=(3, 4))
    block = generate_block(cfg, 3)
    plain = generate_block(small_config(), 3)
    assert np.array_equal(block.seis, plain.seis) and np.array_equal(block.tau, plain.tau)
    assert plain.csd is None and len(block.csd) == cfg.grid.n_slices
    for s in range(cfg.grid.n_slices):
        c = block.csd[s]
        mask, dist = c[("csd_mask", None)], c[("csd_dist", None)]
        assert set(np.unique(mask)) <= {0.0, 1.0} and mask.any()
        assert (dist[mask > 0.5] == 0).all() and (dist[mask < 0.5] > 0).all()
        for S in cfg.grid.scales:
            shown, prompt = c[("csd_prompt", S)], c[("csd_dip", S)]
            assert np.array_equal(shown, mask > 0.5)
            assert np.allclose(prompt[shown], block.labels[s][("dip", S)][shown])
            assert (prompt[~shown] == 0.0).all()
    counts = block.meta["csd_horizons"]
    assert len(counts) == cfg.grid.n_slices and all(1 <= n <= 4 for n in counts)

    noisy = small_config(csd__enabled=True, csd__pick_jitter=(2.0, 2.0), csd__prompt_dropout=0.5,
                         csd__distance="vertical", csd__distance_clip=10.0, csd__distance_normalize=True,
                         csd__prompt_background=-1.0)
    c = generate_block(noisy, 3).csd[0]
    d = c[("csd_dist", None)]
    assert d.min() == 0.0 and d.max() <= 1.0
    shown = c[("csd_prompt", 12)]
    assert shown.sum() < (c[("csd_mask", None)] > 0.5).sum()
    assert (c[("csd_dip", 12)][~shown] == -1.0).all()


def test_csd_folders_are_written_and_inspected(tmp_path):
    from synthetic_generator.checks import inspect_dataset

    cfg = small_config(csd__enabled=True)
    cfg.output.arrays = ["seis", "fault"]
    cfg.output.num_blocks = 1
    generate_dataset(cfg, tmp_path)
    folders = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert folders == ["CsdDipXscale12_75", "CsdDipXscale6_75", "CsdDistance75", "CsdHorizon75", "Fault75", "Seis75"]
    rep = inspect_dataset(tmp_path, sample=6)
    assert {"csd_mask", "csd_dist", "csd_dip6", "csd_dip12"} <= set(rep["folders"]) and not rep["problems"]
    assert rep["finetuner_csd_pairs"] == 6
    data = read_block_from_disk(tmp_path, cfg, 0, 0)
    assert np.array_equal(data[Product("csd_mask")][0], data[Product("csd_mask")][1])


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


def test_run_estimate_follows_blocks_workers_and_bytes():
    cfg = GeneratorConfig()
    cfg.output.num_blocks = 10
    cfg.output.workers = 1
    one = estimate_run_seconds(cfg, 0.5, write_bps=200_000_000)
    cfg.output.num_blocks = 20
    two = estimate_run_seconds(cfg, 0.5, write_bps=200_000_000)
    assert two == pytest.approx(one * 2)
    cfg.output.workers = 4
    parallel = estimate_run_seconds(cfg, 0.5, write_bps=200_000_000)
    assert parallel < two
    cfg.output.dtype = "float32"
    assert estimate_run_seconds(cfg, 0.5, write_bps=200_000_000) < parallel
    assert estimate_run_seconds(cfg, 0.5, n_blocks=0) == 0.0


def test_cpu_weight_tracks_grid_replicas_and_wavelet():
    base = GeneratorConfig()
    weight = block_cpu_weight(base)
    wider = base.copy()
    wider.grid.width *= 2
    assert block_cpu_weight(wider) == pytest.approx(weight * 2)
    fewer = base.copy()
    fewer.grid.n_slices = 3
    fewer.grid.n_replicas = 2
    assert block_cpu_weight(fewer) / weight == pytest.approx(0.414, rel=0.02)
    quick = base.copy()
    quick.seismic.supersample = 1
    mapped = base.copy()
    mapped.seismic.render_mode = "tau-mapping"
    assert block_cpu_weight(quick) < weight
    assert block_cpu_weight(mapped) < weight
    with_csd = base.copy()
    with_csd.csd.enabled = True
    assert block_cpu_weight(with_csd) > weight


def test_write_probe_removes_its_files(tmp_path):
    rate = measure_write_bps(tmp_path, n_files=2)
    assert rate > 0
    assert list(tmp_path.iterdir()) == []


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


@pytest.mark.parametrize("view", ["overview", "arrays", "slices", "replicas", "labels", "sam3", "csd"])
def test_every_preview_view_renders(view):
    from synthetic_generator.preview import fit_tile, render_view

    cfg = small_config()
    block = generate_block(cfg, 2, csd=True)
    image, infos = render_view(block, cfg, view, replica=1, slice_index=2, scale=12, tile=96)
    assert image.width > 96 and image.height > 96
    assert any(i.array is not None for i in infos)
    assert fit_tile(view, cfg, 1200, 700) >= 64


def test_overview_shows_csd_when_enabled():
    from synthetic_generator.preview import render_view

    cfg = small_config()
    block = generate_block(cfg, 2, csd=True)
    _, plain = render_view(block, cfg, "overview", tile=96)
    cfg.csd.enabled = True
    _, with_csd = render_view(block, cfg, "overview", replica=0, slice_index=0, scale=6, tile=96)
    assert [i.label for i in plain] == ["Seis s1 r0", "Seis s1 r0", "Fault", "TauAtT", "dip S=6", "weight S=6"]
    assert [i.label for i in with_csd] == [
        "Seis", "dip S=6", "weight S=6", "CsdDipXscale6", "CsdHorizon", "CsdDistance",
    ]
    _, sam_off = render_view(block, small_config(), "sam3", tile=96)
    assert "CsdHorizon" not in [i.label for i in sam_off]
    _, sam_on = render_view(block, cfg, "sam3", tile=96)
    assert [i.label for i in sam_on].count("CsdHorizon") == 1


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
    monkeypatch.setattr(gui_mod, "measure_write_bps", lambda *args, **kwargs: 200_000_000)
    try:
        root.withdraw()
        gui = gui_mod.GeneratorGUI(root)
        root.update()
        gui._apply_change()
        assert gui.eta_text.get().startswith("Estimated time: about ")
        assert gui.eta_label.cget("style") == "Estimate.TLabel"
        assert "8" in str(gui_mod.ttk.Style(root).lookup("Estimate.TLabel", "font"))
        assert "MB/s" not in gui.summary_text.get()
        assert root.cget("bg") == gui_mod.APP_BG
        cfg, errors = gui.read_config()
        assert errors == [] and cfg.validate() == []
        gui.vars["faults.stroke_radius"][1].set("2.5")
        gui.array_vars["halo"].set(False)
        cfg, errors = gui.read_config()
        assert cfg.faults.stroke_radius[1] == 2.5 and "halo" not in cfg.output.arrays
        tabs = [gui.notebook.tab(t, "text") for t in gui.notebook.tabs()]
        assert tabs == ["Dataset", "Structure", "Faults", "Seismic", "Realism", "Labels", "CSD (In-Progress)",
                        "Data Viewer"]
        menu = gui.root.nametowidget(gui.ft_btn["menu"])
        labels = [menu.entrycget(i, "label") for i in range(menu.index("end") + 1)]
        assert labels == [spec["menu"] for spec in gui_mod.FINETUNE_PRESETS.values() if spec]
        gui.vars["csd.enabled"].set(True)
        gui.vars["csd.connector"].set("step")
        cfg, errors = gui.read_config()
        assert errors == [] and cfg.csd.enabled and cfg.csd.connector == "step"
        gui.vars["faults.count"][0].set("abc")
        assert gui.read_config()[1]
        gui.preset_var.set("Spec: fixed reflections, 2-trace step")
        gui._apply_preset()
        cfg, errors = gui.read_config()
        assert errors == [] and cfg.faults.step_direction == "right" and cfg.structure.slice_drift == (0.0, 0.0)
        assert gui.view_var.get() == "overview"
        gui.preset_var.set("CSD: horizon tracks + dip prompts")
        gui._apply_preset()
        cfg, errors = gui.read_config()
        assert errors == [] and cfg.csd.enabled and gui.view_var.get() == "overview"

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
    monkeypatch.setattr(gui_mod, "measure_write_bps", lambda *args, **kwargs: 200_000_000)
    monkeypatch.setattr(ft, "SETTINGS_FILE", tmp_path / "ft.json")
    launched = []
    monkeypatch.setattr(gui_mod.subprocess, "Popen", lambda *a, **k: launched.append(a[0]))
    data = tmp_path / "data"
    for name in ("Seis75", "Fault75", "TauAtT75", "CsdHorizon75"):
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
        gui.send_to_finetuner("csd")
        saved = json.loads((tmp_path / "ft.json").read_text(encoding="utf-8"))
        assert saved["image_dir"] == str(data / "Seis75") and saved["label_dir"] == str(data / "CsdHorizon75")
        assert ft.PRESETS[saved["preset"]]["task"] == "csd" and saved["prompt"] == "horizon"
        assert ft.LABEL_MODES[saved["label_mode"]] == "binary"
        gui.vars["output.suffix"].set("80")
        for name in ("Seis80", "CsdHorizon80"):
            (data / name).mkdir()
        gui.send_to_finetuner("csd")
        saved = json.loads((tmp_path / "ft.json").read_text(encoding="utf-8"))
        assert saved["image_dir"] == str(data / "Seis80") and saved["label_dir"] == str(data / "CsdHorizon80")
        assert launched and launched[0][1:] == ["-m", "finetuning.finetune_gui"]
    finally:
        if "gui" in locals():
            gui.shutdown()
        root.destroy()


def _write_dat(directory, sid: int, values: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(values, dtype="<f8").tofile(directory / f"{sid}.dat")


def test_scan_dataset_reads_array_folders_and_one_section_folder(tmp_path):
    from synthetic_generator.viewer import (SampleLayout, default_keys, describe_id, load_sample, render_sample,
                                            scan_dataset, step_id)

    root = tmp_path / "set"
    _write_dat(root / "Fault75", 2, np.zeros((4, 4)))
    _write_dat(root / "Seis75", 2, np.linspace(-1.2, 1.2, 16).reshape(4, 4))
    _write_dat(root / "dipXscale12_75", 2, np.full((4, 4), 0.5))
    _write_dat(root / "wtXscale12_75", 2, np.ones((4, 4)))
    _write_dat(root / "wtXStraddleScale60_75", 2, np.zeros((4, 4)))
    _write_dat(root / "CsdHorizon75", 2, np.ones((4, 4)))
    _write_dat(root / "CsdDipXscale12_75", 2, np.zeros((4, 4)))
    (root / "generator_config.json").write_text(json.dumps({"grid": {"n_slices": 3, "n_replicas": 2}}))
    cat = scan_dataset(root)
    assert [(a.kind, a.scale) for a in cat.arrays] == [
        ("seis", None), ("fault", None), ("dip", 12), ("weight", 12), ("straddle", 60),
        ("csd_mask", None), ("csd_dip", 12),
    ]
    assert cat.ids == (2,) and cat.shape == (4, 4) and cat.layout == SampleLayout(3, 2)
    assert default_keys(cat.arrays) == {"seis", "fault"}
    assert describe_id(cat.layout, 2) == "id 2  |  block 0  |  slice 3/3  |  replica 0"
    assert describe_id(None, 4) == "id 4"
    assert step_id((0, 1, 2, 10), 2, 1) == 10
    assert step_id((0, 1, 2, 10), 2, -1) == 1
    assert step_id((0, 1, 2, 10), 0, -1) == 0
    assert step_id((0, 1, 2, 10), 10, 1) == 10
    assert step_id((0, 1, 2, 10), 5, 1) == 10
    assert step_id((0, 1, 2, 10), 5, -1) == 2

    seis = np.linspace(-1.2, 1.2, 16).reshape(4, 4)
    fault = np.zeros((4, 4))
    fault[1, 2] = 1
    _write_dat(root / "Seis75", 2, seis)
    _write_dat(root / "Fault75", 2, fault)
    loaded = load_sample(cat, 2, {"seis", "fault"}, True)
    assert np.allclose(loaded.sections[0][1], seis) and loaded.fault[1, 2] == 1
    plain, infos = render_sample(cat, loaded, 2, "gray", False, 64)
    over, _ = render_sample(cat, loaded, 2, "gray", True, 64)
    assert plain.tobytes() != over.tobytes()
    assert infos[0].label == "Seis" and np.allclose(infos[0].array, seis)

    broken = tmp_path / "broken"
    _write_dat(broken / "Seis75", 1, np.ones((4, 4)))
    (broken / "generator_config.json").write_text("{not json")
    assert scan_dataset(broken).layout is None

    lone = tmp_path / "pack" / "Seis75"
    _write_dat(lone, 3, np.ones((4, 4)))
    (lone.parent / "generator_config.json").write_text(
        json.dumps({"grid": {"n_slices": 6, "n_replicas": 4}}))
    one = scan_dataset(lone)
    assert one.arrays[0].kind == "seis" and one.ids == (3,)
    assert one.layout == SampleLayout(6, 4)
    assert describe_id(one.layout, 3) == "id 3  |  block 0  |  slice 4/6  |  replica 0"

    lines = tmp_path / "lines"
    section = np.linspace(-1.5, 1.5, 64).reshape(8, 8)
    _write_dat(lines, 4, section)
    np.save(lines / "4.npy", np.zeros((8, 8)))
    loose = scan_dataset(lines)
    assert loose.arrays[0].kind == "section" and loose.ids == (4,)
    got = load_sample(loose, 4, {"section"}, False)
    assert np.allclose(got.sections[0][1], section)
    _, tiles = render_sample(loose, got, 4, "gray", False, 48)
    assert tiles[0].label == "Seis"

    rect = tmp_path / "rect"
    rect.mkdir()
    np.save(rect / "8.npy", np.arange(15, dtype=np.float64).reshape(3, 5))
    wide = scan_dataset(rect)
    assert wide.shape == (3, 5)
    wide_sample = load_sample(wide, 8, {"section"}, False)
    assert wide_sample.sections[0][1].shape == (3, 5)

    empty = tmp_path / "empty"
    (empty / "Seis75").mkdir(parents=True)
    assert "Seis75" in scan_dataset(empty).hint

    survey = tmp_path / "survey"
    survey.mkdir()
    (survey / "vol.sgy").write_bytes(b"x")
    assert "SEG-Y" in scan_dataset(survey).hint

    outer = tmp_path / "outer"
    _write_dat(outer / "synthetic_75" / "Seis75", 1, np.zeros((4, 4)))
    assert "synthetic_75" in scan_dataset(outer).hint

    with pytest.raises(FileNotFoundError):
        scan_dataset(tmp_path / "missing")
    text = tmp_path / "file.txt"
    text.write_text("x")
    with pytest.raises(NotADirectoryError):
        scan_dataset(text)


def test_load_sample_rejects_a_file_that_is_not_one_section(tmp_path, monkeypatch):
    import synthetic_generator.viewer as viewer

    monkeypatch.setattr(viewer, "MAX_SECTION_BYTES", 16)
    folder = tmp_path / "Seis75"
    folder.mkdir()
    (folder / "1.dat").write_bytes(b"0123456789abcdef" * 4)
    cat = viewer.scan_dataset(folder)
    sample = viewer.load_sample(cat, 1, {"seis"}, False)
    assert sample.sections == [] and sample.errors


def test_data_viewer_tab_shows_a_folder(monkeypatch, tmp_path):
    root = _tk_root()
    import synthetic_generator.gui as gui_mod

    monkeypatch.setattr(gui_mod, "SETTINGS_FILE", tmp_path / "gui.json")
    monkeypatch.setattr(gui_mod, "measure_write_bps", lambda *args, **kwargs: 200_000_000)
    data = tmp_path / "data"
    _write_dat(data / "Seis75", 0, np.full((8, 8), 0.25))
    _write_dat(data / "Seis75", 1, np.full((8, 8), -0.5))
    fault = np.zeros((8, 8))
    fault[1, 1] = 1.0
    _write_dat(data / "Fault75", 0, fault)
    _write_dat(data / "Fault75", 1, fault)
    (data / "generator_config.json").write_text(json.dumps({"grid": {"n_slices": 3, "n_replicas": 2}}))
    try:
        root.withdraw()
        gui = gui_mod.GeneratorGUI(root)
        root.update()
        tab = next(t for t in gui.notebook.tabs() if gui.notebook.tab(t, "text") == "Data Viewer")
        gui.notebook.select(tab)
        root.update()
        assert gui._disk_mode and gui.viewer_banner.winfo_manager() == "pack"
        gui.viewer_dir.set(str(data))
        gui._view_folder()
        seis = next(info for info in gui.preview_infos if info.label == "Seis")
        assert seis.array[0, 0] == pytest.approx(0.25)
        assert any(info.label == "Fault" for info in gui.preview_infos)
        assert "Seis75" in gui.viewer_status.get() and "slice 1/3" in gui.viewer_place.get()
        assert gui.viewer_status_label.cget("style") == "Hint.TLabel"

        gui._step_viewer_id(1)
        seis = next(info for info in gui.preview_infos if info.label == "Seis")
        assert gui.viewer_id.get() == 1 and seis.array[0, 0] == pytest.approx(-0.5)
        assert "slice 2/3" in gui.viewer_place.get()

        gui._viewer_select("seis")
        assert [info.label for info in gui.preview_infos] == ["Seis"]
        gui._save_settings()
        saved = json.loads((tmp_path / "gui.json").read_text(encoding="utf-8"))
        assert saved["viewer_dir"] == str(data) and saved["viewer_arrays"] == ["seis"] and saved["viewer_id"] == 1

        gui._start_preview = lambda *args, **kwargs: None
        gui.notebook.select(gui.notebook.tabs()[0])
        root.update()
        assert not gui._disk_mode
        assert gui.preview_bar.winfo_manager() == "pack" and gui.viewer_banner.winfo_manager() == ""
    finally:
        if "gui" in locals():
            gui.proc = None
            gui.shutdown()
        root.destroy()
