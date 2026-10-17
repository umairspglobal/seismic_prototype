"""Seismic rendering: impedance stratigraphy -> reflectivity -> bandpass replicas.

The impedance log is defined along tau (geologic time), so every reflector follows
the structural model. Two renderers:

``convolution``  reflectivity is sampled per pixel (supersampled in depth so thin
                 beds do not alias) and convolved down each trace with the
                 wavelet - the standard convolutional model; contacts at faults
                 and unconformities reflect and interfere naturally.
``tau-mapping``  the wavelet is applied along tau and the trace is looked up at
                 every pixel's tau (fast; reflectors stay perfectly sharp at
                 faults).

Each replica uses its own bandpass center. Geology effects (chaotic zones,
lateral gain, fault-zone dimming) are shared by the replicas; noise is drawn per
replica. Every replica is scaled to mean 0 / std 1 over the whole block and
clipped at a level drawn from the clip range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.special import ndtri

from . import sampling
from .config import GeneratorConfig

TAU_STEP = 0.25
GAUSS_BOUND = 1e-6


def gaussianize(x: np.ndarray, bound: float = GAUSS_BOUND) -> np.ndarray:
    """Rank-based map to a standard normal: mean 0, std 1, extremes bounded at +-ndtri(1 - bound)."""
    flat = x.ravel()
    ranks = np.empty(flat.size)
    ranks[np.argsort(flat, kind="stable")] = np.arange(flat.size)
    u = np.clip((ranks + 0.5) / flat.size, bound, 1.0 - bound)
    return ndtri(u).reshape(x.shape)


def normalize(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "gaussianize":
        return gaussianize(x)
    std = x.std()
    return (x - x.mean()) / (std if std > 0 else 1.0)


@dataclass
class Impedance:
    grid: np.ndarray
    log_impedance: np.ndarray
    thickness: float
    memory: float
    facies: list[np.ndarray]


def _ar1(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    innov = rng.standard_normal(n)
    values = np.empty(n)
    values[0] = innov[0]
    scale = np.sqrt(max(1.0 - rho * rho, 1e-6))
    for i in range(1, n):
        values[i] = rho * values[i - 1] + scale * innov[i]
    return values


def make_impedance(rng: np.random.Generator, cfg: GeneratorConfig, tau_lo: float, tau_hi: float,
                   n_facies: int = 2) -> Impedance:
    """Blocky log-impedance along tau, plus ``n_facies`` logs on the same layers for lateral facies change."""
    se = cfg.seismic
    tau_lo, tau_hi = tau_lo - 8.0, tau_hi + 8.0
    grid = np.arange(tau_lo, tau_hi + TAU_STEP, TAU_STEP)
    mean_th = sampling.uniform(rng, se.layer_thickness)
    span = tau_hi - tau_lo
    th = rng.gamma(2.0, mean_th / 2.0, int(span / mean_th * 2) + 16)
    while th.sum() < span + mean_th:
        th = np.concatenate([th, rng.gamma(2.0, mean_th / 2.0, 32)])
    bounds = tau_lo + np.cumsum(th)
    bounds = bounds[bounds < tau_hi]
    rho = sampling.uniform(rng, se.impedance_memory)
    values = _ar1(rng, bounds.size + 1, rho)
    markers = rng.random(values.size) < se.marker_prob
    values[markers] += se.marker_strength * rng.standard_normal(int(markers.sum()))
    envelope = np.ones(bounds.size)
    if se.amplitude_envelope > 0 and bounds.size > 4:
        field = sampling.gaussian_field(rng, (grid.size,), 30.0 / TAU_STEP)
        envelope = np.exp(se.amplitude_envelope * np.interp(bounds, grid, field))
    index = np.searchsorted(bounds, grid, side="right")

    def to_log(vals: np.ndarray) -> np.ndarray:
        steps = np.concatenate([[0.0], np.cumsum(np.diff(vals) * envelope)])
        return steps[index]

    facies = [to_log(_ar1(rng, bounds.size + 1, rho)) for _ in range(n_facies)] if se.lateral_variation > 0 else []
    return Impedance(grid=grid, log_impedance=to_log(values), thickness=mean_th, memory=rho, facies=facies)


def amplitude_spectrum(freqs: np.ndarray, center: float, kind: str, bandwidth: float) -> np.ndarray:
    """Zero-phase amplitude response peaking near ``center`` (cycles/sample)."""
    f = np.abs(freqs)
    if kind == "ricker":
        r = (f / center) ** 2
        return r * np.exp(1.0 - r)
    if kind == "gaussian":
        sigma = max(bandwidth * center / 2.3548, 1e-6)
        low_cut = 1.0 - np.exp(-((f / (0.35 * center)) ** 2))
        return np.exp(-0.5 * ((f - center) / sigma) ** 2) * low_cut
    if kind == "ormsby":
        f1, f2 = center * max(1.0 - bandwidth, 0.05), center * max(1.0 - bandwidth / 2, 0.1)
        f3, f4 = center * (1.0 + bandwidth / 2), center * (1.0 + bandwidth)
        up = np.clip((f - f1) / max(f2 - f1, 1e-9), 0, 1)
        down = np.clip((f4 - f) / max(f4 - f3, 1e-9), 0, 1)
        return np.minimum(up, down) * (f > f1) * (f < f4)
    if kind == "butterworth":
        width = max(bandwidth * center, 1e-6)
        q = (f * f - center * center) / (f * width + 1e-12)
        return 1.0 / np.sqrt(1.0 + q ** 6)
    raise ValueError(f"Unknown wavelet {kind!r}")


def filter_traces(data: np.ndarray, center: float, kind: str, bandwidth: float, phase_deg: float,
                  samples_per_unit: float = 1.0) -> np.ndarray:
    """Bandpass every column of ``data`` (axis 0) with zero padding (no wrap-around)."""
    n = data.shape[0]
    pad = int(np.ceil(3.0 / max(center, 1e-3) * samples_per_unit))
    nfft = next_fast_len(n + 2 * pad)
    spec = rfft(data, n=nfft, axis=0)
    freqs = rfftfreq(nfft, d=1.0 / samples_per_unit)
    response = amplitude_spectrum(freqs, center, kind, bandwidth).astype(np.complex128)
    if phase_deg:
        response *= np.exp(-1j * np.deg2rad(phase_deg))
        response[0] = response[0].real
    shape = (-1,) + (1,) * (data.ndim - 1)
    return irfft(spec * response.reshape(shape), n=nfft, axis=0)[:n]


def band_noise(rng: np.random.Generator, shape: tuple[int, ...], band: tuple[float, float]) -> np.ndarray:
    """Unit-std noise band-limited along axis 0 (cycles/sample) with soft edges."""
    white = rng.standard_normal(shape)
    n = shape[0]
    nfft = next_fast_len(n)
    spec = rfft(white, n=nfft, axis=0)
    f = rfftfreq(nfft)
    lo, hi = band
    taper = 0.02
    resp = sampling.smoothstep((f - (lo - taper)) / (2 * taper)) * sampling.smoothstep(((hi + taper) - f) / (2 * taper))
    out = irfft(spec * resp.reshape((-1,) + (1,) * (len(shape) - 1)), n=nfft, axis=0)[:n]
    std = out.std()
    return out / std if std > 0 else out


def replica_centers(rng: np.random.Generator, cfg: GeneratorConfig) -> list[float]:
    se, R = cfg.seismic, cfg.grid.n_replicas
    if se.center_mode == "random":
        centers = [sampling.log_uniform(rng, se.random_centers) for _ in range(R)]
    else:
        base = list(se.centers)
        centers = [base[i % len(base)] for i in range(R)]
        if se.center_jitter > 0:
            centers = [c * (1.0 + rng.uniform(-se.center_jitter, se.center_jitter)) for c in centers]
        if se.shuffle_centers:
            centers = [centers[i] for i in rng.permutation(R)]
    return [float(np.clip(c, 0.005, 0.45)) for c in centers]


def lowest_center(cfg: GeneratorConfig) -> float:
    se = cfg.seismic
    if se.center_mode == "random":
        return float(se.random_centers[0])
    return float(min(se.centers) * (1.0 - se.center_jitter))


@dataclass
class FaciesField:
    """Smooth weights over (tau, x) per slice that mix the facies logs into the main log."""

    fields: list[np.ndarray]
    lo: float
    step: float
    strength: float

    def at(self, k: int, z: int, tau: np.ndarray) -> np.ndarray:
        F = self.fields[k][z]
        pos = (tau - self.lo) / self.step
        i0 = np.clip(np.floor(pos).astype(np.int64), 0, F.shape[0] - 2)
        w = np.clip(pos - i0, 0.0, 1.0)
        cols = np.broadcast_to(np.arange(F.shape[1]), tau.shape)
        return F[i0, cols] * (1.0 - w) + F[i0 + 1, cols] * w


def make_facies_field(rng: np.random.Generator, cfg: GeneratorConfig, imp: Impedance) -> FaciesField | None:
    se = cfg.seismic
    if se.lateral_variation <= 0 or not imp.facies:
        return None
    step = 4.0
    lo = float(imp.grid[0])
    rows = int(np.ceil((imp.grid[-1] - lo) / step)) + 2
    length = sampling.uniform(rng, se.lateral_length)
    shape = (cfg.grid.n_slices, rows, cfg.grid.width)
    fields = [sampling.gaussian_field(rng, shape, (0.8, 2.5, length)) for _ in imp.facies]
    return FaciesField(fields=fields, lo=lo, step=step, strength=se.lateral_variation)


def reflectivity_section(tau: np.ndarray, imp: Impedance, ss: int, facies: FaciesField | None = None,
                         z: int = 0) -> np.ndarray:
    """Reflection coefficients at ``ss`` sub-samples per row, shape (rows * ss, cols)."""
    rows, cols = tau.shape
    d = np.diff(tau, axis=0)
    ok = (d > 0.05) & (d < 3.0)
    good = np.where(ok, d, 0.0)
    zero_row = np.zeros((1, cols))
    total = np.vstack([zero_row, good]) + np.vstack([good, zero_row])
    count = np.vstack([zero_row, ok]) + np.vstack([ok, zero_row])
    grad = np.where(count > 0, total / np.maximum(count, 1), 1.0)
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    tau_ss = (tau[:, None, :] + offs[None, :, None] * grad[:, None, :]).reshape(rows * ss, cols)
    log_imp = np.interp(tau_ss, imp.grid, imp.log_impedance)
    if facies is not None:
        for k, extra in enumerate(imp.facies):
            log_imp = log_imp + facies.strength * facies.at(k, z, tau_ss) * np.interp(tau_ss, imp.grid, extra)
    refl = np.zeros_like(log_imp)
    refl[:-1] = 0.5 * np.diff(log_imp, axis=0)
    return refl


@dataclass
class SeismicResult:
    seis: np.ndarray
    centers: list[float]
    clips: list[float]
    snrs: list[float]
    phases: list[float]
    meta: dict


def _chaos(rng: np.random.Generator, cfg: GeneratorConfig) -> tuple[np.ndarray | None, int]:
    re = cfg.realism
    N, H, W = cfg.grid.n_slices, cfg.grid.height, cfg.grid.width
    if rng.random() >= re.chaos_prob:
        return None, 0
    count = sampling.integer(rng, re.chaos_count)
    if count <= 0:
        return None, 0
    yy, xx = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
    rough = sampling.gaussian_field(rng, (N, H, W), (1.0, 6.0, 6.0))
    zone = np.zeros((N, H, W))
    for _ in range(count):
        a = sampling.uniform(rng, re.chaos_size)
        b = a * rng.uniform(0.35, 0.9)
        cx, cy = rng.uniform(0, W), rng.uniform(0.08 * H, 0.95 * H)
        ang = rng.uniform(0, np.pi)
        drift = rng.uniform(-1.5, 1.5)
        strength = sampling.uniform(rng, re.chaos_strength)
        edge = max(3.0, 0.25 * min(a, b))
        ca, sa = np.cos(ang), np.sin(ang)
        for z in range(N):
            u = (xx - cx - drift * (z - (N - 1) / 2)) * ca + (yy - cy) * sa
            v = -(xx - cx - drift * (z - (N - 1) / 2)) * sa + (yy - cy) * ca
            r = np.sqrt((u / a) ** 2 + (v / b) ** 2) + 0.12 * rough[z]
            zone[z] = np.maximum(zone[z], strength * sampling.smoothstep((1.0 - r) * min(a, b) / edge))
    return zone, count


def render_block(rng: np.random.Generator, cfg: GeneratorConfig, taus: list[np.ndarray], masks: list[np.ndarray],
                 pad: int) -> SeismicResult:
    """``taus`` are padded sections (H + 2 * pad rows); ``masks`` are the cropped fault masks."""
    se, re = cfg.seismic, cfg.realism
    N, H, W, R = cfg.grid.n_slices, cfg.grid.height, cfg.grid.width, cfg.grid.n_replicas
    lo = min(float(t.min()) for t in taus)
    hi = max(float(t.max()) for t in taus)
    imp = make_impedance(rng, cfg, lo, hi)
    facies = make_facies_field(rng, cfg, imp)
    centers = replica_centers(rng, cfg)
    phases = [sampling.uniform(rng, se.phase) for _ in range(R)]
    ss = max(1, int(se.supersample))

    if se.render_mode == "convolution":
        refl = [reflectivity_section(t, imp, ss, facies, z) for z, t in enumerate(taus)]
    else:
        logs = [imp.log_impedance] + list(imp.facies if facies is not None else [])
        trace_refl = np.zeros((imp.grid.size, len(logs)))
        for k, log in enumerate(logs):
            trace_refl[:-1, k] = 0.5 * np.diff(log)
        mid = imp.grid + 0.5 * TAU_STEP
        weights = None
        if facies is not None:
            weights = [[facies.strength * facies.at(k, z, t[pad:pad + H]) for k in range(len(imp.facies))]
                       for z, t in enumerate(taus)]

    zone, n_zones = _chaos(rng, cfg)
    chaos_white = rng.standard_normal((N, H, W)) if zone is not None else None
    gain_strength = sampling.uniform(rng, re.gain_strength)
    gain = None
    if gain_strength > 0:
        length = sampling.uniform(rng, re.gain_length)
        gain = np.exp(gain_strength * sampling.gaussian_field(rng, (N, H, W), (1.5, length, length)))
    dim_strength = sampling.uniform(rng, re.fault_dimming)
    dimming = None
    if dim_strength > 0 and any(m.any() for m in masks):
        width = max(re.fault_dimming_width, 0.3)
        dimming = np.stack([
            1.0 - dim_strength * np.exp(-0.5 * (ndi.distance_transform_edt(~m) / width) ** 2) if m.any()
            else np.ones((H, W))
            for m in masks
        ])
    statics = None
    if re.trace_jitter > 0:
        statics = re.trace_jitter * sampling.gaussian_field(rng, (N, W), (1.0, 2.0))

    out = np.empty((R, N, H, W))
    clips, snrs, clipped = [], [], []
    for r, (fc, ph) in enumerate(zip(centers, phases)):
        if se.render_mode == "convolution":
            clean = np.stack([
                filter_traces(x, fc, se.wavelet, se.bandwidth, ph, samples_per_unit=ss)
                .reshape(H + 2 * pad, ss, W).mean(axis=1)[pad:pad + H]
                for x in refl
            ])
        else:
            traces = filter_traces(trace_refl, fc, se.wavelet, se.bandwidth, ph, samples_per_unit=1.0 / TAU_STEP)
            sections = []
            for z, t in enumerate(taus):
                tt = t[pad:pad + H]
                sec = np.interp(tt, mid, traces[:, 0])
                if weights is not None:
                    for k, w in enumerate(weights[z]):
                        sec = sec + w * np.interp(tt, mid, traces[:, k + 1])
                sections.append(sec)
            clean = np.stack(sections)
        if statics is not None:
            clean = _apply_statics(clean, statics)
        rms = float(np.sqrt(np.mean(clean ** 2))) or 1.0
        if zone is not None:
            texture = filter_traces(np.moveaxis(chaos_white, 1, 0), fc, se.wavelet, se.bandwidth, ph)
            texture = ndi.gaussian_filter(np.moveaxis(texture, 0, 1), (0.8, 0.0, 2.0))
            texture *= rms * 0.8 / (texture.std() or 1.0)
            clean = clean * (1.0 - zone) + zone * texture
        if gain is not None:
            clean = clean * gain
        if dimming is not None:
            clean = clean * dimming
        snr = sampling.uniform(rng, re.noise_snr)
        snrs.append(snr)
        if snr < 999:
            noise = band_noise(rng, (H, N, W), tuple(re.noise_band))
            noise = np.moveaxis(noise, 0, 1)
            if re.noise_lateral > 0:
                noise = ndi.gaussian_filter1d(noise, re.noise_lateral, axis=2)
                noise /= noise.std() or 1.0
            clean = clean + noise * (np.sqrt(np.mean(clean ** 2)) / snr)
        if re.lateral_smoothing > 0:
            clean = ndi.gaussian_filter1d(clean, re.lateral_smoothing, axis=2)
        if se.normalize_scope == "slice":
            clean = np.stack([normalize(c, se.normalize) for c in clean])
        else:
            clean = normalize(clean, se.normalize)
        if rng.random() < se.clip_prob:
            lo_c, hi_c = se.clip
            clip = float(lo_c + (hi_c - lo_c) * rng.random() ** se.clip_skew)
            clean = np.clip(clean, -clip, clip)
            clipped.append(True)
        else:
            clip = float(np.abs(clean).max())
            clipped.append(False)
        clips.append(clip)
        out[r] = clean

    meta = {
        "centers": [round(c, 5) for c in centers],
        "centers_hz": [round(c * 1000.0 / cfg.grid.sample_interval_ms, 2) for c in centers],
        "phases": [round(p, 2) for p in phases],
        "clips": [round(c, 4) for c in clips],
        "clipped": clipped,
        "snr": [round(s, 3) for s in snrs],
        "layer_thickness": round(imp.thickness, 3),
        "impedance_memory": round(imp.memory, 3),
        "lateral_facies": bool(facies is not None),
        "chaotic_zones": n_zones,
        "gain_strength": round(gain_strength, 3),
        "fault_dimming": round(dim_strength, 3),
    }
    return SeismicResult(seis=out, centers=centers, clips=clips, snrs=snrs, phases=phases, meta=meta)


def _apply_statics(data: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Shift every trace down by ``shifts[z, x]`` samples (Fourier shift, axis 1)."""
    N, H, W = data.shape
    nfft = next_fast_len(2 * H)
    spec = rfft(data, n=nfft, axis=1)
    f = rfftfreq(nfft)
    phase = np.exp(-2j * np.pi * f[None, :, None] * shifts[:, None, :])
    return irfft(spec * phase, n=nfft, axis=1)[:, :H]
