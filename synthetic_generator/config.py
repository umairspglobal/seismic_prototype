"""Every setting of the synthetic generator, presets and JSON round-tripping.

Ranges are ``(low, high)`` tuples and are sampled once per block (or per fault /
replica where noted), so a dataset mixes many geological settings instead of
repeating one. Units: ``samples`` are vertical pixels (tau samples), ``px`` /
``traces`` are horizontal pixels, frequencies are cycles per sample (multiply by
``1000 / sample_interval_ms`` for Hz).
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

ARRAY_KINDS = ("seis", "fault", "tau", "dip", "weight", "straddle", "halo")
WAVELETS = ("ricker", "gaussian", "ormsby", "butterworth")
RENDER_MODES = ("convolution", "tau-mapping")
CENTER_MODES = ("list", "random")
STEP_DIRECTIONS = ("random", "right", "left", "same")
THROW_PROFILES = ("elliptical", "linear", "constant")
DIP_FILLS = ("edge", "zero")
HALO_MODES = ("2d", "horizontal")
NORMALIZE_MODES = ("gaussianize", "standardize")
NORMALIZE_SCOPES = ("block", "slice")
DTYPES = ("float64", "float32")
CSD_KINDS = ("csd_mask", "csd_dip", "csd_dist")
CSD_PLACEMENTS = ("random", "even")
CSD_CONNECTORS = ("random", "line", "smooth", "step")
CSD_DISTANCES = ("euclidean", "vertical")


@dataclass
class OutputConfig:
    out_dir: str = ""
    suffix: str = "75"
    num_blocks: int = 40
    start_id: int = -1
    seed: int = 20261016
    workers: int = 0
    dtype: str = "float64"
    arrays: list[str] = field(default_factory=lambda: list(ARRAY_KINDS))
    overwrite: bool = False


@dataclass
class GridConfig:
    height: int = 224
    width: int = 224
    n_slices: int = 6
    n_replicas: int = 4
    scales: list[int] = field(default_factory=lambda: [12, 36, 60])
    sample_interval_ms: float = 4.0


@dataclass
class StructureConfig:
    t0: tuple[float, float] = (0.0, 35.0)
    fold_count: tuple[int, int] = (1, 5)
    fold_amplitude: tuple[float, float] = (2.0, 34.0)
    fold_width: tuple[float, float] = (8.0, 60.0)
    tilt: tuple[float, float] = (-0.15, 0.15)
    depth_scaling: tuple[float, float] = (0.35, 2.6)
    thickening: tuple[float, float] = (-0.06, 0.06)
    unconformity_prob: float = 0.45
    unconformity_count: tuple[int, int] = (1, 2)
    unconformity_depth: tuple[float, float] = (0.12, 0.88)
    unconformity_relief: tuple[float, float] = (2.0, 14.0)
    unconformity_hiatus: tuple[float, float] = (10.0, 80.0)
    slice_drift: tuple[float, float] = (0.0, 1.5)
    slice_amp_change: float = 0.03


@dataclass
class FaultConfig:
    count: tuple[int, int] = (1, 5)
    dip_median: float = 75.0
    dip_spread: float = 10.0
    dip_limits: tuple[float, float] = (45.0, 89.5)
    p_dip_left: float = 0.5
    segments: tuple[int, int] = (1, 3)
    kink_jitter: float = 4.0
    listric: float = 0.0
    p_through: float = 0.55
    length: tuple[float, float] = (0.45, 1.0)
    stroke_radius: tuple[float, float] = (1.55, 1.85)
    throw: tuple[float, float] = (2.5, 30.0)
    p_reverse: float = 0.05
    throw_profile: str = "elliptical"
    tip_taper: float = 12.0
    drag_prob: float = 0.35
    drag_length: tuple[float, float] = (4.0, 14.0)
    drag_strength: tuple[float, float] = (0.2, 0.7)
    step: tuple[float, float] = (1.0, 3.0)
    step_direction: str = "random"
    step_jitter: float = 0.2
    throw_change: float = 0.03
    min_spacing: float = 10.0


@dataclass
class SeismicConfig:
    render_mode: str = "convolution"
    wavelet: str = "ricker"
    center_mode: str = "list"
    centers: list[float] = field(default_factory=lambda: [0.075, 0.105, 0.14, 0.18])
    center_jitter: float = 0.12
    random_centers: tuple[float, float] = (0.06, 0.20)
    shuffle_centers: bool = True
    bandwidth: float = 0.8
    phase: tuple[float, float] = (0.0, 0.0)
    layer_thickness: tuple[float, float] = (2.0, 6.0)
    impedance_memory: tuple[float, float] = (0.0, 0.7)
    marker_prob: float = 0.05
    marker_strength: float = 2.0
    amplitude_envelope: float = 0.25
    lateral_variation: float = 0.35
    lateral_length: tuple[float, float] = (25.0, 120.0)
    normalize: str = "gaussianize"
    normalize_scope: str = "block"
    clip_prob: float = 0.2
    clip: tuple[float, float] = (2.05, 3.7)
    clip_skew: float = 2.5
    supersample: int = 4


@dataclass
class RealismConfig:
    gain_strength: tuple[float, float] = (0.05, 0.35)
    gain_length: tuple[float, float] = (15.0, 45.0)
    chaos_prob: float = 0.4
    chaos_count: tuple[int, int] = (1, 2)
    chaos_size: tuple[float, float] = (25.0, 95.0)
    chaos_strength: tuple[float, float] = (0.5, 0.95)
    noise_snr: tuple[float, float] = (3.0, 12.0)
    noise_band: tuple[float, float] = (0.04, 0.45)
    noise_lateral: float = 0.8
    fault_dimming: tuple[float, float] = (0.0, 0.5)
    fault_dimming_width: float = 2.0
    lateral_smoothing: float = 0.0
    trace_jitter: float = 0.0


@dataclass
class LabelConfig:
    dip_smoothing: float = 1.0
    dip_fill: str = "edge"
    corridor_pad: int = 2
    partner_pad_left: int = 2
    partner_pad_right: int = 5
    partner_outside: bool = True
    unconformity_rule: bool = True
    halo_sigma_factor: float = 0.5
    halo_quantile: float = 0.99
    halo_mode: str = "2d"


@dataclass
class CsdConfig:
    """Coarse-scale dips: sparse horizon tracks, the dip sampled on them, and the distance to them."""

    enabled: bool = False
    count: tuple[int, int] = (3, 7)
    placement: str = "random"
    depth_range: tuple[float, float] = (0.04, 0.96)
    min_separation: float = 14.0
    same_across_slices: bool = True
    p_partial: float = 0.3
    partial_length: tuple[float, float] = (0.15, 0.8)
    p_stop_at_fault: float = 0.1
    connect_prob: float = 0.85
    connector: str = "random"
    skip_stroke: bool = False
    gap_prob: float = 0.2
    gap_count: tuple[int, int] = (1, 2)
    gap_length: tuple[float, float] = (3.0, 18.0)
    thickness: tuple[int, int] = (1, 2)
    pick_jitter: tuple[float, float] = (0.0, 0.0)
    jitter_length: float = 12.0
    prompt_noise: tuple[float, float] = (0.0, 0.0)
    prompt_dropout: float = 0.0
    prompt_horizon_dropout: float = 0.0
    prompt_weighted: bool = False
    prompt_background: float = 0.0
    distance: str = "euclidean"
    distance_clip: float = 0.0
    distance_normalize: bool = False


@dataclass
class GeneratorConfig:
    output: OutputConfig = field(default_factory=OutputConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    faults: FaultConfig = field(default_factory=FaultConfig)
    seismic: SeismicConfig = field(default_factory=SeismicConfig)
    realism: RealismConfig = field(default_factory=RealismConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    csd: CsdConfig = field(default_factory=CsdConfig)

    @property
    def ids_per_block(self) -> int:
        return self.grid.n_slices * self.grid.n_replicas

    def copy(self) -> "GeneratorConfig":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = SCHEMA_VERSION
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratorConfig":
        cfg = cls()
        for section in fields(cls):
            raw = data.get(section.name)
            if not isinstance(raw, dict):
                continue
            target = getattr(cfg, section.name)
            for f in fields(target):
                if f.name in raw:
                    setattr(target, f.name, _coerce(getattr(target, f.name), raw[f.name]))
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GeneratorConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def get(self, path: str) -> Any:
        section, name = path.split(".", 1)
        return getattr(getattr(self, section), name)

    def set(self, path: str, value: Any) -> None:
        section, name = path.split(".", 1)
        target = getattr(self, section)
        setattr(target, name, _coerce(getattr(target, name), value))

    def validate(self) -> list[str]:
        """Human-readable problems; an empty list means the config can be generated."""
        errors: list[str] = []
        g, o, st, fa, se, re, la = (self.grid, self.output, self.structure, self.faults,
                                     self.seismic, self.realism, self.labels)

        def rng(name: str, value, lo=None, hi=None, integer=False) -> None:
            a, b = value
            if a > b:
                errors.append(f"{name}: minimum {a} is larger than maximum {b}.")
            if lo is not None and a < lo:
                errors.append(f"{name}: minimum must be at least {lo}.")
            if hi is not None and b > hi:
                errors.append(f"{name}: maximum must be at most {hi}.")
            if integer and (int(a) != a or int(b) != b):
                errors.append(f"{name}: values must be whole numbers.")

        if not 32 <= g.height <= 2048 or not 32 <= g.width <= 2048:
            errors.append("Image size must be between 32 and 2048 pixels.")
        if g.height != g.width:
            errors.append("Sections must be square (height = width): the Quick Fine-Tuner infers the shape "
                          "from the file size.")
        if not 1 <= g.n_slices <= 64:
            errors.append("Slices per block must be between 1 and 64.")
        if not 1 <= g.n_replicas <= 16:
            errors.append("Bandpass replicas must be between 1 and 16.")
        if not g.scales:
            errors.append("At least one dip scale S is required.")
        for s in g.scales:
            if not 1 <= s <= g.width - 3:
                errors.append(f"Scale S={s} must be between 1 and width - 3 ({g.width - 3}).")
        if len(set(g.scales)) != len(g.scales):
            errors.append("Dip scales must be unique.")
        if g.sample_interval_ms <= 0:
            errors.append("Sample interval must be positive.")

        if o.num_blocks < 1:
            errors.append("Number of blocks must be at least 1.")
        if o.start_id < -1:
            errors.append("Start id must be -1 (continue) or a non-negative id.")
        if o.dtype not in DTYPES:
            errors.append(f"Output dtype must be one of {DTYPES}.")
        if o.suffix and not o.suffix.replace("_", "").isalnum():
            errors.append("Folder suffix may only contain letters, digits and underscores.")
        unknown = [a for a in o.arrays if a not in ARRAY_KINDS]
        if unknown:
            errors.append(f"Unknown array kinds: {unknown}.")
        if not o.arrays:
            errors.append("Select at least one array to write.")
        if o.workers < 0:
            errors.append("Workers must be 0 (auto) or positive.")

        rng("Start tau t0", st.t0)
        rng("Fold count", st.fold_count, 0, 12, integer=True)
        rng("Fold amplitude", st.fold_amplitude, 0)
        rng("Fold width", st.fold_width, 2)
        rng("Regional tilt", st.tilt, -1.0, 1.0)
        rng("Fold depth scaling", st.depth_scaling, 0)
        rng("Layer thickening", st.thickening, -0.4, 0.4)
        rng("Unconformity count", st.unconformity_count, 0, 4, integer=True)
        rng("Unconformity depth", st.unconformity_depth, 0.0, 1.0)
        rng("Unconformity relief", st.unconformity_relief, 0)
        rng("Unconformity hiatus", st.unconformity_hiatus, 0)
        rng("Reflector drift per slice", st.slice_drift, 0)
        _prob(errors, "Unconformity probability", st.unconformity_prob)

        rng("Fault count", fa.count, 0, 16, integer=True)
        rng("Dip limits", fa.dip_limits, 5.0, 90.0)
        if not fa.dip_limits[0] <= fa.dip_median <= fa.dip_limits[1]:
            errors.append("Median fault dip must lie inside the dip limits.")
        rng("Fault segments", fa.segments, 1, 8, integer=True)
        rng("Partial fault length", fa.length, 0.02, 3.0)
        rng("Stroke radius", fa.stroke_radius, 0.3, 12.0)
        rng("Throw", fa.throw, 0)
        rng("Fault step per slice", fa.step, 0)
        rng("Drag length", fa.drag_length, 0.5)
        rng("Drag strength", fa.drag_strength, 0, 1)
        for name, p in (("Dip-left probability", fa.p_dip_left), ("Through-going probability", fa.p_through),
                        ("Reverse-fault probability", fa.p_reverse), ("Drag probability", fa.drag_prob)):
            _prob(errors, name, p)
        if fa.throw_profile not in THROW_PROFILES:
            errors.append(f"Throw profile must be one of {THROW_PROFILES}.")
        if fa.step_direction not in STEP_DIRECTIONS:
            errors.append(f"Step direction must be one of {STEP_DIRECTIONS}.")
        for name, v in (("Dip spread", fa.dip_spread), ("Kink jitter", fa.kink_jitter),
                        ("Step jitter", fa.step_jitter), ("Throw change per slice", fa.throw_change),
                        ("Fold amplitude change per slice", st.slice_amp_change), ("Tip taper", fa.tip_taper),
                        ("Minimum spacing", fa.min_spacing), ("Noise lateral smoothing", re.noise_lateral),
                        ("Migration-like smoothing", re.lateral_smoothing), ("Trace jitter", re.trace_jitter),
                        ("Dip smoothing", la.dip_smoothing), ("Marker-bed strength", se.marker_strength),
                        ("Bright/dim intervals", se.amplitude_envelope)):
            if v < 0:
                errors.append(f"{name} cannot be negative.")

        if se.render_mode not in RENDER_MODES:
            errors.append(f"Render mode must be one of {RENDER_MODES}.")
        if se.wavelet not in WAVELETS:
            errors.append(f"Wavelet must be one of {WAVELETS}.")
        if se.center_mode not in CENTER_MODES:
            errors.append(f"Center mode must be one of {CENTER_MODES}.")
        if se.center_mode == "list":
            if not se.centers:
                errors.append("Give at least one bandpass center.")
            for c in se.centers:
                if not 0.005 <= c <= 0.45:
                    errors.append(f"Bandpass center {c} must be between 0.005 and 0.45 cycles/sample.")
        rng("Random bandpass centers", se.random_centers, 0.005, 0.45)
        if not 0 <= se.center_jitter < 0.9:
            errors.append("Center jitter must be between 0 and 0.9.")
        if not 0.1 <= se.bandwidth <= 2.0:
            errors.append("Relative bandwidth must be between 0.1 and 2.")
        rng("Wavelet phase", se.phase, -180, 180)
        rng("Layer thickness", se.layer_thickness, 0.3)
        rng("Impedance memory", se.impedance_memory, 0.0, 0.99)
        _prob(errors, "Marker-bed probability", se.marker_prob)
        if not 0 <= se.lateral_variation <= 3:
            errors.append("Lateral facies variation must be between 0 and 3.")
        rng("Lateral facies length", se.lateral_length, 1)
        rng("Clip level", se.clip, 0.1)
        _prob(errors, "Clip probability", se.clip_prob)
        if se.clip_skew <= 0:
            errors.append("Clip skew must be positive (1 = uniform).")
        if se.normalize not in NORMALIZE_MODES:
            errors.append(f"Normalization must be one of {NORMALIZE_MODES}.")
        if se.normalize_scope not in NORMALIZE_SCOPES:
            errors.append(f"Normalization scope must be one of {NORMALIZE_SCOPES}.")
        if not 1 <= se.supersample <= 16:
            errors.append("Supersampling must be between 1 and 16.")

        rng("Gain strength", re.gain_strength, 0, 3)
        rng("Gain correlation length", re.gain_length, 1)
        _prob(errors, "Chaotic-zone probability", re.chaos_prob)
        rng("Chaotic-zone count", re.chaos_count, 0, 10, integer=True)
        rng("Chaotic-zone size", re.chaos_size, 2)
        rng("Chaotic-zone strength", re.chaos_strength, 0, 1)
        rng("Signal-to-noise ratio", re.noise_snr, 0.1)
        rng("Noise band", re.noise_band, 0.0, 0.5)
        rng("Fault dimming", re.fault_dimming, 0, 1)

        if la.dip_fill not in DIP_FILLS:
            errors.append(f"Dip fill must be one of {DIP_FILLS}.")
        if la.halo_mode not in HALO_MODES:
            errors.append(f"Halo mode must be one of {HALO_MODES}.")
        if not 0.5 <= la.halo_quantile <= 1.0:
            errors.append("Halo normalization quantile must be between 0.5 and 1.")
        if la.halo_sigma_factor <= 0:
            errors.append("Halo sigma factor must be positive.")
        for name, v in (("Corridor pad", la.corridor_pad), ("Partner pad (left)", la.partner_pad_left),
                        ("Partner pad (right)", la.partner_pad_right)):
            if v < 0:
                errors.append(f"{name} cannot be negative.")

        cs = self.csd
        rng("CSD horizon count", cs.count, 0, 40, integer=True)
        rng("CSD depth range", cs.depth_range, 0.0, 1.0)
        rng("CSD partial length", cs.partial_length, 0.02, 1.0)
        rng("CSD gap count", cs.gap_count, 0, 10, integer=True)
        rng("CSD gap length", cs.gap_length, 0)
        rng("CSD line thickness", cs.thickness, 1, 9, integer=True)
        rng("CSD pick jitter", cs.pick_jitter, 0)
        rng("CSD prompt noise", cs.prompt_noise, 0)
        for name, p in (("CSD partial probability", cs.p_partial), ("CSD stop-at-fault probability", cs.p_stop_at_fault),
                        ("CSD connect probability", cs.connect_prob), ("CSD gap probability", cs.gap_prob),
                        ("CSD prompt pixel dropout", cs.prompt_dropout),
                        ("CSD prompt horizon dropout", cs.prompt_horizon_dropout)):
            _prob(errors, name, p)
        if cs.placement not in CSD_PLACEMENTS:
            errors.append(f"CSD placement must be one of {CSD_PLACEMENTS}.")
        if cs.connector not in CSD_CONNECTORS:
            errors.append(f"CSD connector must be one of {CSD_CONNECTORS}.")
        if cs.distance not in CSD_DISTANCES:
            errors.append(f"CSD distance must be one of {CSD_DISTANCES}.")
        for name, v in (("CSD minimum separation", cs.min_separation), ("CSD jitter length", cs.jitter_length),
                        ("CSD distance clip", cs.distance_clip)):
            if v < 0:
                errors.append(f"{name} cannot be negative.")
        return errors


def _prob(errors: list[str], name: str, p: float) -> None:
    if not 0.0 <= p <= 1.0:
        errors.append(f"{name} must be between 0 and 1.")


def _coerce(current: Any, value: Any) -> Any:
    """Convert JSON / widget values to the type of the existing field value."""
    if isinstance(current, tuple):
        lo, hi = value
        kind = type(current[0])
        return (kind(round(float(lo))) if kind is int else kind(lo),
                kind(round(float(hi))) if kind is int else kind(hi))
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(current, int):
        return int(round(float(value)))
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        if isinstance(value, str):
            value = [v for v in value.replace(";", ",").split(",") if v.strip()]
        kind = type(current[0]) if current else str
        if kind is str:
            return [str(v).strip() for v in value]
        if kind is int:
            return [int(round(float(v))) for v in value]
        return [float(v) for v in value]
    return value


def _spec(cfg: GeneratorConfig) -> None:
    """The literal 8-step description: smooth tau, steep faults, fixed reflections."""
    st, fa, se, re = cfg.structure, cfg.faults, cfg.seismic, cfg.realism
    st.unconformity_prob = 0.0
    st.depth_scaling = (1.0, 1.0)
    st.thickening = (0.0, 0.0)
    st.slice_drift = (0.0, 0.0)
    st.slice_amp_change = 0.0
    fa.count = (1, 3)
    fa.dip_median = 72.0
    fa.dip_spread = 8.0
    fa.p_dip_left = 1.0
    fa.stroke_radius = (2.0, 3.0)
    fa.throw = (2.0, 6.0)
    fa.p_reverse = 0.0
    fa.throw_profile = "constant"
    fa.drag_prob = 0.0
    fa.step = (2.0, 2.0)
    fa.step_direction = "right"
    fa.step_jitter = 0.0
    fa.throw_change = 0.0
    se.shuffle_centers = False
    se.center_jitter = 0.0
    se.lateral_variation = 0.0
    se.normalize = "standardize"
    se.clip_prob = 1.0
    se.clip = (2.1, 4.7)
    se.clip_skew = 1.0
    re.gain_strength = (0.0, 0.0)
    re.chaos_prob = 0.0
    re.noise_snr = (1000.0, 1000.0)
    re.fault_dimming = (0.0, 0.0)


def _clean(cfg: GeneratorConfig) -> None:
    _spec(cfg)
    cfg.faults.count = (1, 2)
    cfg.faults.stroke_radius = (1.6, 1.95)
    cfg.faults.throw = (4.0, 10.0)


def _complex(cfg: GeneratorConfig) -> None:
    st, fa, re = cfg.structure, cfg.faults, cfg.realism
    st.fold_amplitude = (4.0, 26.0)
    st.fold_count = (2, 6)
    st.unconformity_prob = 0.7
    st.unconformity_count = (1, 3)
    fa.count = (2, 6)
    fa.segments = (1, 4)
    fa.listric = 4.0
    fa.p_through = 0.45
    re.chaos_prob = 0.55
    re.chaos_count = (1, 3)
    re.fault_dimming = (0.1, 0.6)


def _noisy(cfg: GeneratorConfig) -> None:
    re = cfg.realism
    re.noise_snr = (1.2, 4.0)
    re.gain_strength = (0.3, 0.7)
    re.chaos_prob = 0.6
    re.lateral_smoothing = 0.6
    re.trace_jitter = 0.35


def _low_freq(cfg: GeneratorConfig) -> None:
    cfg.seismic.centers = [0.045, 0.06, 0.075, 0.09]
    cfg.seismic.random_centers = (0.04, 0.10)


def _high_freq(cfg: GeneratorConfig) -> None:
    cfg.seismic.centers = [0.14, 0.17, 0.20, 0.24]
    cfg.seismic.random_centers = (0.13, 0.26)
    cfg.seismic.layer_thickness = (1.2, 3.5)


def _faults_only(cfg: GeneratorConfig) -> None:
    cfg.output.arrays = ["seis", "fault"]


def _csd(cfg: GeneratorConfig) -> None:
    """Noisy field-like data plus horizon tracks; writes Seis, dip and weight (the CSD targets) and the CSD folders."""
    _noisy(cfg)
    cfg.output.arrays = ["seis", "dip", "weight"]
    cs = cfg.csd
    cs.enabled = True
    cs.count = (3, 8)
    cs.placement = "random"
    cs.depth_range = (0.04, 0.96)
    cs.min_separation = 14.0
    cs.same_across_slices = True
    cs.p_partial = 0.3
    cs.partial_length = (0.2, 0.8)
    cs.gap_prob = 0.25
    cs.gap_count = (1, 2)
    cs.gap_length = (3.0, 15.0)
    cs.thickness = (1, 2)
    cs.connect_prob = 0.85
    cs.connector = "random"
    cs.p_stop_at_fault = 0.1
    cs.skip_stroke = False
    cs.pick_jitter = (0.0, 1.0)
    cs.jitter_length = 15.0
    cs.prompt_noise = (0.0, 0.5)
    cs.prompt_dropout = 0.1
    cs.prompt_horizon_dropout = 0.1
    cs.prompt_weighted = False
    cs.prompt_background = 0.0
    cs.distance = "euclidean"
    cs.distance_clip = 0.0
    cs.distance_normalize = False


PRESETS: dict[str, Any] = {
    "Noisy field-like data": _noisy,
    "CSD: horizon tracks + dip prompts": _csd,
    "Spec: fixed reflections, 2-trace step": _spec,
    "Clean and simple (debugging)": _clean,
    "Complex structure": _complex,
    "Low frequency (poor resolution)": _low_freq,
    "High frequency (sharp)": _high_freq,
    "Faults only (Seis + Fault folders)": _faults_only,
}
DEFAULT_PRESET = "Noisy field-like data"


def default_config() -> GeneratorConfig:
    """What a fresh GUI, Reset and the command line start from."""
    return preset(DEFAULT_PRESET)


def preset(name: str, base: GeneratorConfig | None = None) -> GeneratorConfig:
    """A fresh config for ``name``; output and CSD settings are kept from ``base`` if given.

    Only the CSD preset turns CSD writing on; every other preset turns it off.
    """
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; choose from {list(PRESETS)}")
    cfg = GeneratorConfig()
    if base is not None:
        cfg.output = copy.deepcopy(base.output)
        cfg.output.arrays = list(ARRAY_KINDS)
        cfg.csd = copy.deepcopy(base.csd)
        cfg.csd.enabled = False
    PRESETS[name](cfg)
    return cfg
