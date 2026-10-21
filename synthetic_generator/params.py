"""Every GUI control: config path, label, widget kind, limits and help text."""

from __future__ import annotations

from dataclasses import dataclass

from .config import (CENTER_MODES, CSD_CONNECTORS, CSD_DISTANCES, CSD_PLACEMENTS, DIP_FILLS, DTYPES, HALO_MODES,
                     NORMALIZE_MODES, NORMALIZE_SCOPES, RENDER_MODES, STEP_DIRECTIONS, THROW_PROFILES, WAVELETS)


@dataclass(frozen=True)
class Param:
    path: str
    label: str
    kind: str
    lo: float | None = None
    hi: float | None = None
    step: float = 1.0
    choices: tuple[str, ...] = ()
    help: str = ""


P = Param

TABS: list[tuple[str, list[tuple[str, list[Param]]]]] = [
    ("Dataset", [
        ("Output", [
            P("output.out_dir", "Output folder", "dir",
              help="One sub-folder per array (Seis75, Fault75, ...) is created here. Point the Quick Fine-Tuner's "
                   "Data folder at it."),
            P("output.suffix", "Folder suffix", "text",
              help="Folders are named Seis{suffix}, Fault{suffix}, dipXscale12_{suffix}, ... '75' matches "
                   "training_10_7_26 and the fine-tuner presets."),
            P("output.num_blocks", "Blocks", "int", 1, 1_000_000,
              help="One block = one geological model = slices x replicas ids (24 by default)."),
            P("output.start_id", "First id", "int", -1, 100_000_000,
              help="-1 continues after the ids already in the folder (block aligned). Use an explicit id to resume "
                   "or overwrite a range."),
            P("output.seed", "Random seed", "int", 0, 2**31 - 1,
              help="Block k always uses (seed, k), so the same seed and ids reproduce the same data."),
            P("output.workers", "Worker processes", "int", 0, 64, help="0 = automatic (cores - 1, at most 8)."),
            P("output.dtype", "File type", "choice", choices=DTYPES,
              help="float64 little-endian like the reference (and 1.dat). float32 halves the disk use; the "
                   "fine-tuner reads both."),
            P("output.overwrite", "Rewrite existing blocks", "bool",
              help="Off: complete blocks already on disk are skipped (resume). On: rewrite them."),
            P("output.arrays", "Arrays to write", "arrays",
              help="Seismic, fault mask, tau and the labels at every scale. Seis + Fault is all the fault "
                   "fine-tuner needs."),
        ]),
        ("Section grid", [
            P("grid.height", "Section size (px, square)", "int", 32, 2048, 16,
              help="Height = width. The Quick Fine-Tuner reads square sections and infers the size from the "
                   "file; 224 matches training_10_7_26."),
            P("grid.n_slices", "Slices per block", "int", 1, 64,
              help="Consecutive sections cut from one model; the fault trace steps sideways between them."),
            P("grid.n_replicas", "Bandpass replicas", "int", 1, 16,
              help="Each slice is rendered once per bandpass center; only Seis differs between replicas."),
            P("grid.scales", "Dip scales S", "list",
              help="Comma-separated trace offsets. Each S gives dipXscaleS, wtXscaleS, wtXStraddleScaleS and "
                   "wtXHaloScaleS."),
            P("grid.sample_interval_ms", "Sample interval (ms)", "float", 0.1, 100, 0.5,
              help="Only used to show frequencies in Hz."),
        ]),
    ]),
    ("Structure", [
        ("Reflectors: tau = t0 + y + u(x)", [
            P("structure.t0", "Tau at the top (t0)", "range_float", 0, 1000, 1),
            P("structure.fold_count", "Fold bumps per package", "range_int", 0, 12),
            P("structure.fold_amplitude", "Fold amplitude (samples)", "range_float", 0, 200, 1,
              help="Height of each Gaussian fold bump; signs are random (anticlines and synclines)."),
            P("structure.fold_width", "Fold width sigma (traces)", "range_float", 2, 1000, 2),
            P("structure.tilt", "Regional tilt (samples/trace)", "range_float", -1, 1, 0.01),
            P("structure.depth_scaling", "Fold growth with depth", "range_float", 0, 10, 0.05,
              help="Fold amplitude at the bottom divided by the top (log-uniform). 1 = same everywhere."),
            P("structure.thickening", "Lateral thickening", "range_float", -0.4, 0.4, 0.01,
              help="Layers thicken or thin across the section (wedges)."),
        ]),
        ("Unconformities", [
            P("structure.unconformity_prob", "Probability per block", "prob"),
            P("structure.unconformity_count", "Count when present", "range_int", 0, 4),
            P("structure.unconformity_depth", "Depth (fraction of height)", "range_float", 0, 1, 0.02),
            P("structure.unconformity_relief", "Erosion relief (samples)", "range_float", 0, 100, 1,
              help="Waviness of the erosion surface, including channel-like incisions."),
            P("structure.unconformity_hiatus", "Missing time (tau)", "range_float", 0, 500, 1,
              help="Tau gap across the unconformity; the package above has its own folding."),
        ]),
        ("Between slices", [
            P("structure.slice_drift", "Reflector drift (traces/slice)", "range_float", 0, 20, 0.1,
              help="0 keeps the flat reflections fixed across the six slices (the literal spec)."),
            P("structure.slice_amp_change", "Fold amplitude change per slice", "float", 0, 0.5, 0.01),
        ]),
    ]),
    ("Faults", [
        ("Count and geometry", [
            P("faults.count", "Faults per block", "range_int", 0, 16),
            P("faults.dip_median", "Median dip (deg from horizontal)", "float", 5, 90, 1,
              help="The spec says 72; measured the same way, training_10_7_26 is about 75."),
            P("faults.dip_spread", "Dip spread (std, deg)", "float", 0, 45, 0.5),
            P("faults.dip_limits", "Dip limits (deg)", "range_float", 5, 90, 1),
            P("faults.p_dip_left", "Share dipping left", "prob",
              help="A fault dipping left has its footwall on the right, so its throw sits on the right edge "
                   "of the stroke. 1.0 = always (spec); 0.5 = both ways (reference)."),
            P("faults.segments", "Polyline segments", "range_int", 1, 8),
            P("faults.kink_jitter", "Kink jitter (deg)", "float", 0, 30, 0.5),
            P("faults.listric", "Listric flattening (deg)", "float", 0, 40, 0.5,
              help="Dip decrease from the top segment to the bottom segment."),
            P("faults.p_through", "Through-going share", "prob",
              help="Faults that cross the whole section; the rest have tips inside it."),
            P("faults.length", "Partial fault length (x height)", "range_float", 0.02, 3, 0.05),
            P("faults.min_spacing", "Minimum spacing (traces)", "float", 0, 200, 1),
        ]),
        ("Stroke (the mask)", [
            P("faults.stroke_radius", "Stroke radius (px)", "range_float", 0.3, 12, 0.05,
              help="Pixels within this distance of the fault trace are 1 in Fault75. 1.55-1.85 reproduces the "
                   "3-4 px strokes of 1.dat; 2-3 is the spec's wording (4-6 px)."),
        ]),
        ("Throw", [
            P("faults.throw", "Max throw (tau samples)", "range_float", 0, 200, 0.5,
              help="Log-uniform. Applied to the footwall beyond the stroke edge; the stroke keeps the "
                   "hanging-wall values."),
            P("faults.p_reverse", "Reverse-fault share", "prob"),
            P("faults.throw_profile", "Throw profile along the fault", "choice", choices=THROW_PROFILES),
            P("faults.tip_taper", "Tip taper (px)", "float", 0, 200, 1),
            P("faults.drag_prob", "Drag probability", "prob",
              help="Reflectors bend into the fault and the offset shrinks near it."),
            P("faults.drag_length", "Drag length (px)", "range_float", 0.5, 100, 0.5),
            P("faults.drag_strength", "Drag strength", "range_float", 0, 1, 0.05),
        ]),
        ("Across slices", [
            P("faults.step", "Fault step (traces/slice)", "range_float", 0, 30, 0.1,
              help="How far each fault trace moves between consecutive slices (spec: about 2)."),
            P("faults.step_direction", "Step direction", "choice", choices=STEP_DIRECTIONS,
              help="right = spec; random = each fault picks a side (reference); same = one side per block."),
            P("faults.step_jitter", "Step jitter (traces)", "float", 0, 10, 0.05),
            P("faults.throw_change", "Throw change per slice", "float", 0, 0.5, 0.01),
        ]),
    ]),
    ("Seismic", [
        ("Wavelet and bandpass replicas", [
            P("seismic.render_mode", "Renderer", "choice", choices=RENDER_MODES,
              help="convolution: reflectivity per pixel convolved down each trace (physical). tau-mapping: "
                   "wavelet applied along tau (faster, perfectly sharp at faults)."),
            P("seismic.wavelet", "Wavelet / bandpass", "choice", choices=WAVELETS),
            P("seismic.center_mode", "Center frequencies", "choice", choices=CENTER_MODES,
              help="list: one center per replica from the list (with jitter). random: log-uniform draw per "
                   "replica."),
            P("seismic.centers", "Bandpass centers (cycles/sample)", "list",
              help="One per replica. 0.075/0.105/0.14/0.18 c/s = 19/26/35/45 Hz at 4 ms."),
            P("seismic.center_jitter", "Center jitter (+/- fraction)", "float", 0, 0.89, 0.01),
            P("seismic.random_centers", "Random center range", "range_float", 0.005, 0.45, 0.005),
            P("seismic.shuffle_centers", "Shuffle centers between replicas", "bool"),
            P("seismic.bandwidth", "Relative bandwidth", "float", 0.1, 2, 0.05,
              help="Gaussian / Ormsby / Butterworth only (Ricker has a fixed shape)."),
            P("seismic.phase", "Wavelet phase (deg)", "range_float", -180, 180, 5),
        ]),
        ("Reflectivity", [
            P("seismic.layer_thickness", "Mean layer thickness (samples)", "range_float", 0.3, 100, 0.1),
            P("seismic.impedance_memory", "Impedance memory (AR1)", "range_float", 0, 0.99, 0.05,
              help="Higher = blockier logs with fewer strong contrasts."),
            P("seismic.marker_prob", "Marker-bed probability", "prob"),
            P("seismic.marker_strength", "Marker-bed strength", "float", 0, 20, 0.1),
            P("seismic.amplitude_envelope", "Bright/dim intervals", "float", 0, 3, 0.05),
            P("seismic.lateral_variation", "Lateral facies variation", "float", 0, 3, 0.05,
              help="Reflection strength (and polarity) changes along each layer."),
            P("seismic.lateral_length", "Facies length (traces)", "range_float", 1, 2000, 5),
            P("seismic.supersample", "Depth supersampling", "int", 1, 16),
        ]),
        ("Scaling and clipping", [
            P("seismic.normalize", "Normalization", "choice", choices=NORMALIZE_MODES,
              help="gaussianize: rank-map to N(0,1) per replica like training_10_7_26 (max about 4.7). "
                   "standardize: subtract mean, divide by std (spec wording)."),
            P("seismic.normalize_scope", "Normalize over", "choice", choices=NORMALIZE_SCOPES,
              help="block = all slices of a replica together (reference); slice = each slice alone."),
            P("seismic.clip_prob", "Clip probability", "prob",
              help="Reference: about 20% of replicas are clipped; spec: always."),
            P("seismic.clip", "Clip level (+/-)", "range_float", 0.1, 20, 0.05),
            P("seismic.clip_skew", "Clip skew (1 = uniform)", "float", 0.1, 10, 0.1,
              help=">1 favours low clip levels (reference: mostly 2.1-2.8)."),
        ]),
    ]),
    ("Realism", [
        ("Amplitude", [
            P("realism.gain_strength", "Lateral gain strength", "range_float", 0, 3, 0.05),
            P("realism.gain_length", "Gain correlation (px)", "range_float", 1, 1000, 1),
            P("realism.fault_dimming", "Fault-zone dimming", "range_float", 0, 1, 0.05),
            P("realism.fault_dimming_width", "Dimming width (px)", "float", 0.1, 50, 0.1),
        ]),
        ("Chaotic / transparent zones", [
            P("realism.chaos_prob", "Probability per block", "prob",
              help="Blobs where layering fades into random texture (mass-transport, gas, salt-like)."),
            P("realism.chaos_count", "Zones", "range_int", 0, 10),
            P("realism.chaos_size", "Size (px)", "range_float", 2, 1000, 1),
            P("realism.chaos_strength", "Strength", "range_float", 0, 1, 0.05),
        ]),
        ("Noise and acquisition", [
            P("realism.noise_snr", "Signal-to-noise (amplitude)", "range_float", 0.1, 1000, 0.5,
              help="RMS signal / RMS noise, per replica. 1000 = clean."),
            P("realism.noise_band", "Noise band (cycles/sample)", "range_float", 0, 0.5, 0.01),
            P("realism.noise_lateral", "Noise lateral smoothing (px)", "float", 0, 20, 0.1),
            P("realism.lateral_smoothing", "Migration-like smoothing (px)", "float", 0, 20, 0.1),
            P("realism.trace_jitter", "Trace static jitter (samples)", "float", 0, 10, 0.05),
        ]),
    ]),
    ("Labels", [
        ("Dip  (tau[y,x] - tau[y,x+S])", [
            P("labels.dip_smoothing", "Smoothing sigma (px)", "float", 0, 20, 0.1),
            P("labels.dip_fill", "Columns without a partner", "choice", choices=DIP_FILLS,
              help="edge repeats the last valid column (reference); zero writes 0."),
        ]),
        ("Scale weight (loss mask)", [
            P("labels.corridor_pad", "Corridor around stroke (px)", "int", 0, 50),
            P("labels.partner_pad_left", "Partner corridor left (px)", "int", 0, 50),
            P("labels.partner_pad_right", "Partner corridor right (px)", "int", 0, 50,
              help="With S = 12, pads 2 / 2 / 5 leave the one-pixel notch 3-4 columns before the stroke."),
            P("labels.partner_outside", "Zero where the partner reflector leaves the section", "bool"),
            P("labels.unconformity_rule", "Zero where x and x+S are in different packages", "bool"),
        ]),
        ("Halo", [
            P("labels.halo_sigma_factor", "Blur sigma (x S)", "float", 0.05, 5, 0.05,
              help="Gaussian blur of the stroke with radius S."),
            P("labels.halo_quantile", "Peak normalization quantile", "float", 0.5, 1, 0.005,
              help="0.99 = scaled by the 99th percentile and clipped, so it peaks at 1 (reference)."),
            P("labels.halo_mode", "Blur", "choice", choices=HALO_MODES),
        ]),
    ]),
    ("CSD (In-Progress)", [
        ("Coarse-scale dips", [
            P("csd.enabled", "Write CSD arrays", "bool",
              help="Adds CsdHorizon (target mask), CsdDipXscaleS (prompt: dip on the horizons, one per scale) and "
                   "CsdDistance (distance to the nearest horizon). The CSD preview view works either way."),
        ]),
        ("Horizons", [
            P("csd.count", "Horizons per section", "range_int", 0, 40),
            P("csd.placement", "Placement", "choice", choices=CSD_PLACEMENTS,
              help="random: depths drawn with a minimum separation. even: evenly spaced with a little jitter."),
            P("csd.depth_range", "Depth range (fraction of height)", "range_float", 0, 1, 0.02),
            P("csd.min_separation", "Minimum separation (px)", "float", 0, 200, 1,
              help="Random placement only; fewer horizons are drawn if they do not fit."),
            P("csd.same_across_slices", "Same horizons in every slice", "bool",
              help="On: every slice of the block tracks the same reflectors (tau levels), like picks in a 3D "
                   "volume. Off: each slice picks its own."),
        ]),
        ("Track extent", [
            P("csd.p_partial", "Partial-track share", "prob",
              help="Share of horizons tracked over only part of the section (a picker that stopped)."),
            P("csd.partial_length", "Partial length (fraction of width)", "range_float", 0.02, 1, 0.05),
            P("csd.gap_prob", "Gap probability per horizon", "prob",
              help="Columns where the pick is missing (dropouts)."),
            P("csd.gap_count", "Gaps per horizon", "range_int", 0, 10),
            P("csd.gap_length", "Gap length (px)", "range_float", 0, 200, 1),
            P("csd.thickness", "Line thickness (px)", "range_int", 1, 9),
        ]),
        ("Crossing faults", [
            P("csd.connect_prob", "Connect across a fault", "prob",
              help="The track steps down (or up) by the throw at a fault. Connected: the step is drawn, so the "
                   "horizon dips down along the fault and continues. Otherwise it is left open."),
            P("csd.connector", "Connector shape", "choice", choices=CSD_CONNECTORS,
              help="line: straight ramp across the stroke. smooth: S-curve (drag-like). step: vertical drop in "
                   "the middle of the stroke. random: one shape per horizon."),
            P("csd.p_stop_at_fault", "Stop at a fault", "prob",
              help="Chance that tracking ends at each fault it reaches (growing outward from a seed column)."),
            P("csd.skip_stroke", "Leave the fault stroke empty", "bool",
              help="Removes horizon pixels that fall on the fault stroke (also from the connector)."),
        ]),
        ("Prompt (horizon dips)", [
            P("csd.pick_jitter", "Pick jitter (px)", "range_float", 0, 20, 0.1,
              help="Smooth vertical error of the prompt picks (an imperfect interpreter). The target mask stays "
                   "exact."),
            P("csd.jitter_length", "Jitter correlation (px)", "float", 0.5, 500, 1),
            P("csd.prompt_noise", "Dip noise (tau samples, std)", "range_float", 0, 20, 0.05),
            P("csd.prompt_dropout", "Pixel dropout", "prob", help="Random prompt pixels removed (sparser input)."),
            P("csd.prompt_horizon_dropout", "Horizon dropout", "prob",
              help="Horizons kept in the target but missing from the prompt."),
            P("csd.prompt_weighted", "Only where the scale weight is 1", "bool"),
            P("csd.prompt_background", "Background value", "float", -100, 100, 0.5),
        ]),
        ("Distance map", [
            P("csd.distance", "Metric", "choice", choices=CSD_DISTANCES,
              help="euclidean: 2D distance to the nearest horizon pixel. vertical: along the trace only."),
            P("csd.distance_clip", "Clip (px, 0 = off)", "float", 0, 2048, 1),
            P("csd.distance_normalize", "Normalize to 0..1", "bool",
              help="Divides by the clip value (or by the section size when the clip is off)."),
        ]),
    ]),
]


def all_params() -> list[Param]:
    return [p for _, sections in TABS for _, params in sections for p in params]
