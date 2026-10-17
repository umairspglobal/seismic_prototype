"""Structural model of one block: relative geologic time (tau) and faults.

A block is a small 3D volume seen as ``n_slices`` consecutive sections. The
stratigraphy is a stack of packages (oldest at the bottom) separated by erosional
unconformities; inside a package ``tau = A + y + u(x, y)`` with ``u`` a sum of
fold bumps, a regional tilt, depth-dependent fold growth and lateral thickening.
Everything is analytic, so tau can be evaluated at any (x, y) - which is how the
faults are applied: each fault is a vertical-throw coordinate transform. Faults
are applied youngest first, so older fault traces get offset by younger faults,
exactly like the tau field.

A fault is a polyline that is monotonic in depth. Pixels within ``radius`` of it
form the stroke (the binary label). The footwall side moves by the throw while
the stroke keeps the hanging-wall values, so the tau jump sits on the footwall
edge of the stroke - the right edge for a fault dipping left, which is the
reference dataset's convention ("throw applied on the right edge of the stroke").
Across slices each fault trace steps sideways by its own amount, so the six
sections are a consistent pseudo-3D sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import sampling
from .config import FaultConfig, GeneratorConfig, StructureConfig


@dataclass
class Fold:
    amp: float
    center: float
    width: float
    drift: float
    rate: float


@dataclass
class Package:
    folds: list[Fold]
    tilt: float
    depth_scale: float
    thickening: float
    offset: float = 0.0

    def tau(self, X: np.ndarray, Y: np.ndarray, z: float, H: int, W: int) -> np.ndarray:
        u = np.zeros_like(X, dtype=np.float64)
        for f in self.folds:
            amp = f.amp * (1.0 + f.rate * z)
            u += amp * np.exp(-0.5 * ((X - f.center - f.drift * z) / f.width) ** 2)
        depth_gain = 1.0 + (self.depth_scale - 1.0) * (Y / H)
        stretch = 1.0 + self.thickening * (X - 0.5 * W) / W
        return self.offset + Y * stretch + u * depth_gain + self.tilt * (X - 0.5 * W)


@dataclass
class Surface:
    """Erosional unconformity: everything above ``depth_at`` belongs to a younger package."""

    depth: float
    waves: list[tuple[float, float, float]]
    incisions: list[tuple[float, float, float]]
    drift: float

    def depth_at(self, X: np.ndarray, z: float) -> np.ndarray:
        x = X - self.drift * z
        d = np.full_like(X, self.depth, dtype=np.float64)
        for amp, wavelength, phase in self.waves:
            d += amp * np.sin(2.0 * math.pi * x / wavelength + phase)
        for depth, center, width in self.incisions:
            d += depth * np.exp(-0.5 * ((x - center) / width) ** 2)
        return d


@dataclass
class Stratigraphy:
    packages: list[Package]
    surfaces: list[Surface]
    H: int
    W: int
    shift: float = 0.0

    def tau(self, X: np.ndarray, Y: np.ndarray, z: float) -> np.ndarray:
        out = self.packages[0].tau(X, Y, z, self.H, self.W)
        for pkg, surf in zip(self.packages[1:], self.surfaces):
            above = Y < surf.depth_at(X, z)
            if above.any():
                out = np.where(above, pkg.tau(X, Y, z, self.H, self.W), out)
        return out + self.shift

    def package_index(self, X: np.ndarray, Y: np.ndarray, z: float) -> np.ndarray:
        """Which package (0 = oldest) each point belongs to."""
        out = np.zeros(X.shape, dtype=np.int8)
        for j, surf in enumerate(self.surfaces, start=1):
            out[Y < surf.depth_at(X, z)] = j
        return out


@dataclass
class Fault:
    xs: np.ndarray
    ys: np.ndarray
    dip_dir: int
    radius: float
    throw: float
    profile: str
    peak: float
    taper: float
    offsets: np.ndarray
    throw_rate: float
    drag: float
    drag_strength: float
    order: int
    dip: float
    through: bool
    meta: dict = field(default_factory=dict)

    @property
    def footwall_sign(self) -> int:
        # Dipping left (x shrinks with depth) puts the hanging wall on the left.
        return -self.dip_dir

    def vertices(self, z: int) -> tuple[np.ndarray, np.ndarray]:
        return self.xs + self.offsets[z], self.ys

    def throw_at(self, Y: np.ndarray, z: int, zc: float) -> np.ndarray:
        y0, y1 = float(self.ys[0]), float(self.ys[-1])
        s = np.clip((Y - y0) / max(y1 - y0, 1e-6), 0.0, 1.0)
        if self.profile == "elliptical":
            w = np.where(s < self.peak, self.peak, 1.0 - self.peak)
            shape = np.sqrt(np.clip(1.0 - ((s - self.peak) / np.maximum(w, 1e-6)) ** 2, 0.0, 1.0))
        elif self.profile == "linear":
            shape = 1.0 - s if self.peak < 0.5 else s
        else:
            shape = np.ones_like(s)
        if self.taper > 0:
            tip = np.minimum(Y - y0, y1 - Y)
            shape = shape * sampling.smoothstep(tip / self.taper)
        return self.throw * (1.0 + self.throw_rate * (z - zc)) * shape


@dataclass
class SliceGeometry:
    tau: np.ndarray
    mask: np.ndarray
    strokes: list[np.ndarray]
    offsets: list[np.ndarray]
    inside: list[np.ndarray]
    package: np.ndarray


@dataclass
class BlockGeology:
    strat: Stratigraphy
    faults: list[Fault]
    slices: list[SliceGeometry]
    pad: int
    meta: dict


def polyline_distance(xs: np.ndarray, ys: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Euclidean distance from every (X, Y) to the polyline (round caps at the ends)."""
    best = np.full(X.shape, np.inf)
    for i in range(len(xs) - 1):
        ax, ay, bx, by = xs[i], ys[i], xs[i + 1], ys[i + 1]
        vx, vy = bx - ax, by - ay
        length2 = vx * vx + vy * vy
        if length2 <= 1e-12:
            continue
        t = np.clip(((X - ax) * vx + (Y - ay) * vy) / length2, 0.0, 1.0)
        dx = X - (ax + t * vx)
        dy = Y - (ay + t * vy)
        np.minimum(best, dx * dx + dy * dy, out=best)
    return np.sqrt(best)


def _make_package(rng: np.random.Generator, st: StructureConfig, H: int, W: int) -> Package:
    folds = []
    for _ in range(sampling.integer(rng, st.fold_count)):
        folds.append(Fold(
            amp=sampling.uniform(rng, st.fold_amplitude) * (1 if rng.random() < 0.5 else -1),
            center=rng.uniform(-0.15 * W, 1.15 * W),
            width=sampling.uniform(rng, st.fold_width),
            drift=sampling.uniform(rng, st.slice_drift) * (1 if rng.random() < 0.5 else -1),
            rate=rng.uniform(-st.slice_amp_change, st.slice_amp_change),
        ))
    return Package(
        folds=folds,
        tilt=sampling.uniform(rng, st.tilt),
        depth_scale=sampling.log_uniform(rng, st.depth_scaling),
        thickening=sampling.uniform(rng, st.thickening),
    )


def _make_surface(rng: np.random.Generator, st: StructureConfig, H: int, W: int, depth: float) -> Surface:
    relief = sampling.uniform(rng, st.unconformity_relief)
    waves = []
    for _ in range(int(rng.integers(1, 3))):
        waves.append((relief * rng.uniform(0.3, 0.7), rng.uniform(0.4, 2.5) * W, rng.uniform(0, 2 * math.pi)))
    incisions = []
    for _ in range(int(rng.integers(0, 3))):
        incisions.append((relief * rng.uniform(0.4, 1.4), rng.uniform(0, W), rng.uniform(4, 22)))
    drift = sampling.uniform(rng, st.slice_drift) * (1 if rng.random() < 0.5 else -1)
    return Surface(depth=depth, waves=waves, incisions=incisions, drift=drift)


def make_stratigraphy(rng: np.random.Generator, cfg: GeneratorConfig, pad: int) -> tuple[Stratigraphy, dict]:
    st, H, W, N = cfg.structure, cfg.grid.height, cfg.grid.width, cfg.grid.n_slices
    n_unc = sampling.integer(rng, st.unconformity_count) if rng.random() < st.unconformity_prob else 0
    depths = sorted((sampling.uniform(rng, st.unconformity_depth) * H for _ in range(n_unc)), reverse=True)
    packages = [_make_package(rng, st, H, W) for _ in range(n_unc + 1)]
    surfaces = [_make_surface(rng, st, H, W, d) for d in depths]
    strat = Stratigraphy(packages=packages, surfaces=[], H=H, W=W)

    # Offsets are chosen on a grid spanning every slice so the stack stays in order
    # (younger above older, with a hiatus) and identical across the six sections.
    xs = np.arange(-8, W + 8, 2.0)
    zs = np.arange(N, dtype=float)
    hiatuses = []
    for j, (pkg, surf) in enumerate(zip(packages[1:], surfaces), start=1):
        gaps = []
        for z in zs:
            E = surf.depth_at(xs, z)
            below = strat.tau(xs, E + 0.5, z)
            mine = pkg.tau(xs, E - 0.5, z, H, W)
            gaps.append(np.min(below - mine))
        hiatus = sampling.uniform(rng, st.unconformity_hiatus)
        pkg.offset = float(min(gaps) - hiatus)
        hiatuses.append(hiatus)
        strat.surfaces.append(surf)

    Xg, Yg = np.meshgrid(np.arange(W, dtype=float), np.array([0.0]))
    top = min(float(strat.tau(Xg, Yg, z).min()) for z in zs)
    strat.shift = sampling.uniform(rng, st.t0) - top
    meta = {
        "unconformities": n_unc,
        "unconformity_depths": [round(d, 1) for d in depths],
        "hiatus": [round(h, 1) for h in hiatuses],
        "fold_amplitude_max": round(max((abs(f.amp) for p in packages for f in p.folds), default=0.0), 2),
    }
    return strat, meta


def _fault_x_at(xs: np.ndarray, ys: np.ndarray, y: float) -> float:
    return float(np.interp(y, ys, xs))


def make_faults(rng: np.random.Generator, cfg: GeneratorConfig) -> list[Fault]:
    fa: FaultConfig = cfg.faults
    H, W, N = cfg.grid.height, cfg.grid.width, cfg.grid.n_slices
    zc = (N - 1) / 2.0
    target = sampling.integer(rng, fa.count)
    block_sign = 1 if rng.random() < 0.5 else -1
    faults: list[Fault] = []
    attempts = 0
    while len(faults) < target and attempts < 60 * max(target, 1):
        attempts += 1
        dip = float(np.clip(rng.normal(fa.dip_median, fa.dip_spread), *fa.dip_limits))
        dip_dir = -1 if rng.random() < fa.p_dip_left else 1
        through = rng.random() < fa.p_through
        if through:
            y0, y1 = -rng.uniform(8, 60), H + rng.uniform(8, 60)
        else:
            length = sampling.uniform(rng, fa.length) * H
            lo, hi = sorted((0.3 * length, H - 0.3 * length))
            yc = rng.uniform(lo, hi) if hi > lo else 0.5 * H
            y0, y1 = yc - 0.5 * length, yc + 0.5 * length
        nseg = sampling.integer(rng, fa.segments)
        inner = np.sort(rng.uniform(0.2, 0.8, nseg - 1)) if nseg > 1 else np.array([])
        ys = np.concatenate([[y0], y0 + inner * (y1 - y0), [y1]])
        angles = []
        for i in range(nseg):
            frac = i / (nseg - 1) if nseg > 1 else 0.5
            a = dip + fa.listric * (0.5 - frac) + rng.normal(0.0, fa.kink_jitter)
            angles.append(float(np.clip(a, *fa.dip_limits)))
        dxs = [dip_dir * (ys[i + 1] - ys[i]) / math.tan(math.radians(angles[i])) for i in range(nseg)]
        xs = np.concatenate([[0.0], np.cumsum(dxs)])
        y_mid = float(np.clip(0.5 * (y0 + y1), 0, H - 1))
        x_mid = rng.uniform(0.04 * W, 0.96 * W)
        xs = xs - _fault_x_at(xs, ys, y_mid) + x_mid

        magnitude = sampling.uniform(rng, fa.step)
        if fa.step_direction == "right":
            sign = 1
        elif fa.step_direction == "left":
            sign = -1
        elif fa.step_direction == "same":
            sign = block_sign
        else:
            sign = 1 if rng.random() < 0.5 else -1
        steps = sign * magnitude + rng.normal(0.0, fa.step_jitter, N)
        steps[0] = 0.0
        offsets = np.cumsum(steps)
        offsets -= np.interp(zc, np.arange(N), offsets)

        visible = 0
        for z in (0, N - 1, int(round(zc))):
            yy = np.arange(max(0, int(y0)), min(H, int(y1) + 1))
            xx = np.interp(yy, ys, xs + offsets[z])
            visible = max(visible, int(np.sum((xx >= 0) & (xx < W))))
        if visible < min(24, H // 4):
            continue
        too_close = False
        for other in faults:
            lo, hi = max(y0, other.ys[0], 0), min(y1, other.ys[-1], H - 1)
            if hi <= lo:
                continue
            ym = 0.5 * (lo + hi)
            if abs(_fault_x_at(xs, ys, ym) - _fault_x_at(other.xs, other.ys, ym)) < fa.min_spacing:
                too_close = True
                break
        if too_close:
            continue

        throw = sampling.log_uniform(rng, fa.throw)
        if rng.random() < fa.p_reverse:
            throw = -throw
        drag, drag_strength = 0.0, 0.0
        if rng.random() < fa.drag_prob:
            drag = sampling.uniform(rng, fa.drag_length)
            drag_strength = sampling.uniform(rng, fa.drag_strength)
        faults.append(Fault(
            xs=xs, ys=ys, dip_dir=dip_dir,
            radius=sampling.uniform(rng, fa.stroke_radius),
            throw=throw, profile=fa.throw_profile,
            peak=float(rng.uniform(0.25, 0.75)),
            taper=fa.tip_taper,
            offsets=offsets,
            throw_rate=float(rng.uniform(-fa.throw_change, fa.throw_change)),
            drag=drag, drag_strength=drag_strength, order=0, dip=dip, through=through,
            meta={
                "dip_deg": round(dip, 2), "dips": "left" if dip_dir < 0 else "right",
                "segments": nseg, "through": through,
                "radius": None, "throw": None, "step": round(float(sign * magnitude), 3),
                "drag": round(drag, 2), "drag_strength": round(drag_strength, 3),
            },
        ))
    order = rng.permutation(len(faults))
    for f, o in zip(faults, order):
        f.order = int(o)
        f.meta["radius"] = round(f.radius, 3)
        f.meta["throw"] = round(f.throw, 3)
        f.meta["order"] = int(o)
    return faults


def build_slice(strat: Stratigraphy, faults: list[Fault], z: int, cfg: GeneratorConfig, pad: int) -> SliceGeometry:
    H, W = cfg.grid.height, cfg.grid.width
    zc = (cfg.grid.n_slices - 1) / 2.0
    y = np.arange(-pad, H + pad, dtype=np.float64)
    x = np.arange(W, dtype=np.float64)
    Y, X = np.meshgrid(y, x, indexing="ij")
    Ys = Y.copy()
    strokes, offsets, inside = [], [], []
    for f in sorted(faults, key=lambda f: -f.order):
        xs, ys = f.vertices(z)
        xf = np.interp(Ys, ys, xs)
        within = (Ys >= ys[0]) & (Ys <= ys[-1])
        dx = X - xf
        dist = polyline_distance(xs, ys, X, Ys)
        stroke = dist <= f.radius
        side = dx * f.footwall_sign
        footwall = within & (side > 0) & ~stroke
        if f.drag > 0 and f.drag_strength > 0:
            # Normal drag: the offset shrinks toward the fault and both walls bend into it.
            moved = within.copy()
            T = f.throw_at(Ys[moved], z, zc)
            bend = 0.5 * f.drag_strength * np.exp(-np.maximum(dist[moved] - f.radius, 0.0) / f.drag)
            Ys[moved] += np.where(footwall[moved], T * (1.0 - bend), T * bend)
        elif footwall.any():
            Ys[footwall] += f.throw_at(Ys[footwall], z, zc)
        strokes.append(stroke)
        offsets.append(dx)
        inside.append(within)
    tau = strat.tau(X, Ys, z)
    mask = np.zeros_like(tau, dtype=bool)
    for s in strokes:
        mask |= s
    return SliceGeometry(tau=tau, mask=mask, strokes=strokes, offsets=offsets, inside=inside,
                         package=strat.package_index(X, Ys, z))


def crop(geom: SliceGeometry, pad: int, H: int) -> SliceGeometry:
    cut = slice(pad, pad + H)
    return SliceGeometry(
        tau=geom.tau[cut], mask=geom.mask[cut],
        strokes=[s[cut] for s in geom.strokes],
        offsets=[o[cut] for o in geom.offsets],
        inside=[i[cut] for i in geom.inside],
        package=geom.package[cut],
    )
