"""Coarse-scale dips (CSD): sparse horizon tracks, the dip sampled on them, and the distance to them.

A horizon is one tau level (one reflector). In every column the track sits where
tau crosses that level, so it follows the folds, steps down where it crosses a
fault (the footwall edge carries the throw) and then continues on the other
side. Arrays per section:

csd_mask        1 on the tracked horizons (the target), 0 elsewhere.
csd_dip (S)     the prompt: ``dip`` at scale S on the (optionally jittered, thinned)
                tracks, ``prompt_background`` elsewhere.
csd_dist        distance in px to the nearest horizon pixel (euclidean or vertical),
                optionally clipped and normalised.

A track grows from a seed column to both sides. A fault crossing is bridged by a
connector (line, smooth S-curve or step) with ``connect_prob``, left open
otherwise, and ends the track with ``p_stop_at_fault``. Columns where the level is
missing (eroded at an unconformity, or outside the section) leave a gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from . import sampling
from .config import CSD_CONNECTORS, CsdConfig

CONNECTOR_SHAPES = tuple(c for c in CSD_CONNECTORS if c != "random")


@dataclass
class HorizonPlan:
    depth: float
    window: tuple[float, float]
    seed_x: float
    gaps: list[tuple[float, float]]
    thickness: int
    connector: str
    seed: int
    in_prompt: bool
    level: float | None = None


@dataclass
class TrackedHorizon:
    pieces: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


def plan_horizons(rng: np.random.Generator, cs: CsdConfig, H: int, W: int) -> list[HorizonPlan]:
    n = sampling.integer(rng, cs.count)
    lo, hi = cs.depth_range
    sep = cs.min_separation / max(H, 1)
    if cs.placement == "even" and n > 0:
        step = (hi - lo) / n
        depths = [lo + step * (k + 0.5 + rng.uniform(-0.25, 0.25)) for k in range(n)]
    else:
        depths: list[float] = []
        for _ in range(n):
            for _attempt in range(60):
                d = sampling.uniform(rng, (lo, hi))
                if all(abs(d - o) >= sep for o in depths):
                    depths.append(d)
                    break
    plans = []
    for d in sorted(depths):
        if rng.random() < cs.p_partial:
            length = sampling.uniform(rng, cs.partial_length) * W
            x0 = rng.uniform(0.0, max(W - length, 0.0))
            window = (x0, x0 + length)
        else:
            window = (0.0, float(W))
        seed_x = rng.uniform(*window) if window[1] > window[0] else window[0]
        gaps = []
        if rng.random() < cs.gap_prob:
            for _ in range(sampling.integer(rng, cs.gap_count)):
                gaps.append((rng.uniform(0, W), sampling.uniform(rng, cs.gap_length)))
        connector = cs.connector if cs.connector != "random" else str(rng.choice(CONNECTOR_SHAPES))
        plans.append(HorizonPlan(
            depth=float(d), window=window, seed_x=float(seed_x), gaps=gaps,
            thickness=sampling.integer(rng, cs.thickness), connector=connector,
            seed=int(rng.integers(0, 2**31 - 1)), in_prompt=bool(rng.random() >= cs.prompt_horizon_dropout),
        ))
    return plans


def level_at_depth(tau: np.ndarray, depth: float) -> float:
    """Tau level whose iso-line passes near row ``depth * (H - 1)`` across the section."""
    H = tau.shape[0]
    row = float(np.clip(depth, 0.0, 1.0)) * (H - 1)
    r0 = int(np.floor(row))
    r1 = min(r0 + 1, H - 1)
    t = row - r0
    return float(np.median((1 - t) * tau[r0] + t * tau[r1]))


def pick_columns(tau_max: np.ndarray, level: float, gap_tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel row of ``level`` in every column, and whether that column has a pick."""
    H, W = tau_max.shape
    i = (tau_max < level).sum(axis=0)
    ic = np.clip(i, 1, H - 1)
    cols = np.arange(W)
    t0, t1 = tau_max[ic - 1, cols], tau_max[ic, cols]
    d = t1 - t0
    y = (ic - 1) + np.clip((level - t0) / np.maximum(d, 1e-9), 0.0, 1.0)
    valid = (i > 0) & (i < H) & (d <= gap_tol)
    return y, valid


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of every run of True."""
    padded = np.concatenate([[False], flags, [False]]).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1))


def _connector(xa: float, ya: float, xb: float, yb: float, shape: str) -> tuple[np.ndarray, np.ndarray]:
    """Bridge across a fault; the end points are included (the caller drops them)."""
    if shape == "step":
        xm = 0.5 * (xa + xb)
        return np.array([xa, xm, xm, xb]), np.array([ya, ya, yb, yb])
    t = np.linspace(0.0, 1.0, max(3, int(np.ceil(xb - xa)) * 4 + 1))
    f = sampling.smoothstep(t) if shape == "smooth" else t
    return xa + t * (xb - xa), ya + f * (yb - ya)


def track_horizon(tau_max: np.ndarray, mask: np.ndarray, plan: HorizonPlan, cs: CsdConfig,
                  gap_tol: float) -> TrackedHorizon:
    """Polylines (x, y) of one horizon: runs between faults plus the connectors that bridge them."""
    H, W = mask.shape
    y, valid = pick_columns(tau_max, plan.level, gap_tol)
    cols = np.arange(W)
    rows = np.clip(np.round(y).astype(int), 0, H - 1)
    on_stroke = valid & mask[rows, cols]
    allowed = (cols >= plan.window[0]) & (cols <= plan.window[1])
    for start, length in plan.gaps:
        allowed &= ~((cols >= round(start)) & (cols < round(start + length)))
    runs = _runs(valid & ~on_stroke & allowed)
    if not runs:
        return TrackedHorizon()
    rng = np.random.default_rng(plan.seed)

    seed = plan.seed_x
    start = int(np.argmin([0 if a <= seed <= b else min(abs(seed - a), abs(seed - b)) for a, b in runs]))
    # links[k] is how run k joins run k + 1: "connect", "open" or None (the track ended there).
    links: list[str | None] = [None] * (len(runs) - 1)
    keep = {start}

    def decide(k: int) -> bool:
        a_end, b_start = runs[k][1], runs[k + 1][0]
        between = slice(a_end + 1, b_start)
        crossing = bool(on_stroke[between].any()) and bool(allowed[between].all())
        if crossing:
            if rng.random() < cs.p_stop_at_fault:
                return False
            links[k] = "connect" if rng.random() < cs.connect_prob else "open"
        else:
            links[k] = "open"
        return True

    for k in range(start, len(runs) - 1):
        if not decide(k):
            break
        keep.add(k + 1)
    for k in range(start - 1, -1, -1):
        if not decide(k):
            break
        keep.add(k)

    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    cur_x: list[np.ndarray] = []
    cur_y: list[np.ndarray] = []

    def flush() -> None:
        if cur_x:
            pieces.append((np.concatenate(cur_x), np.concatenate(cur_y)))
            cur_x.clear()
            cur_y.clear()

    order = sorted(keep)
    for idx, k in enumerate(order):
        a, b = runs[k]
        xs = np.arange(a, b + 1, dtype=np.float64)
        cur_x.append(xs)
        cur_y.append(y[a:b + 1])
        nxt = order[idx + 1] if idx + 1 < len(order) else None
        if nxt == k + 1 and links[k] == "connect":
            b_start = runs[k + 1][0]
            cx, cy = _connector(float(b), float(y[b]), float(b_start), float(y[b_start]), plan.connector)
            cur_x.append(cx[1:-1])
            cur_y.append(cy[1:-1])
        else:
            flush()
    flush()
    return TrackedHorizon(pieces=pieces)


def rasterize(pieces: list[tuple[np.ndarray, np.ndarray]], shape: tuple[int, int], thickness: int,
              y_offset=None) -> np.ndarray:
    """Draw polylines densely (no holes on steep steps) with a vertical line thickness."""
    H, W = shape
    out = np.zeros(shape, dtype=bool)
    for px, py in pieces:
        if y_offset is not None:
            py = py + np.interp(px, np.arange(W), y_offset)
        if px.size == 1:
            xs, ys = px, py
        else:
            dx, dy = np.diff(px), np.diff(py)
            n = np.maximum(1, np.ceil(np.maximum(np.abs(dx), np.abs(dy)) * 2).astype(int))
            seg = np.repeat(np.arange(n.size), n)
            frac = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            t = frac / n[seg]
            xs = np.concatenate([px[seg] + t * dx[seg], px[-1:]])
            ys = np.concatenate([py[seg] + t * dy[seg], py[-1:]])
        cols = np.round(xs).astype(int)
        base = np.round(ys - (thickness - 1) / 2.0).astype(int)
        for k in range(thickness):
            r = base + k
            ok = (cols >= 0) & (cols < W) & (r >= 0) & (r < H)
            out[r[ok], cols[ok]] = True
    return out


def distance_map(mask: np.ndarray, cs: CsdConfig) -> np.ndarray:
    H, W = mask.shape
    far = float(cs.distance_clip) if cs.distance_clip > 0 else float(max(H, W))
    if not mask.any():
        dist = np.full(mask.shape, far)
    elif cs.distance == "vertical":
        rows = np.arange(H, dtype=np.float64)[:, None] * np.ones((1, W))
        above = np.maximum.accumulate(np.where(mask, rows, -np.inf), axis=0)
        below = np.flipud(np.minimum.accumulate(np.flipud(np.where(mask, rows, np.inf)), axis=0))
        dist = np.minimum(rows - above, below - rows)
        dist[~np.isfinite(dist)] = far
    else:
        dist = ndi.distance_transform_edt(~mask)
    if cs.distance_clip > 0:
        dist = np.minimum(dist, cs.distance_clip)
    if cs.distance_normalize:
        dist = dist / far
    return dist.astype(np.float64)


def section_csd(tau: np.ndarray, mask: np.ndarray, plans: list[HorizonPlan], labels: dict, scales: list[int],
                cs: CsdConfig, rng: np.random.Generator) -> tuple[dict, int]:
    """CSD arrays of one section plus the number of horizons that left a track."""
    H, W = tau.shape
    tau_max = np.maximum.accumulate(tau, axis=0)
    dy = np.diff(tau, axis=0)
    gap_tol = max(3.0, 4.0 * float(np.median(dy[dy > 0])) if (dy > 0).any() else 3.0)
    stroke = mask > 0.5
    target = np.zeros((H, W), dtype=bool)
    prompt = np.zeros((H, W), dtype=bool)
    drawn = 0
    for plan in plans:
        if plan.level is None:
            plan.level = level_at_depth(tau, plan.depth)
        track = track_horizon(tau_max, stroke, plan, cs, gap_tol)
        if not track.pieces:
            continue
        line = rasterize(track.pieces, (H, W), plan.thickness)
        jitter = None
        amp = sampling.uniform(rng, cs.pick_jitter)
        if amp > 0:
            jitter = amp * sampling.gaussian_field(rng, (W,), max(cs.jitter_length, 0.5))
        shown = rasterize(track.pieces, (H, W), plan.thickness, jitter) if jitter is not None else line.copy()
        if cs.skip_stroke:
            line &= ~stroke
            shown &= ~stroke
        if line.any():
            drawn += 1
        target |= line
        if plan.in_prompt:
            prompt |= shown
    if cs.prompt_dropout > 0:
        prompt &= rng.random((H, W)) >= cs.prompt_dropout
    noise_std = sampling.uniform(rng, cs.prompt_noise)
    noise = rng.standard_normal((H, W)) * noise_std if noise_std > 0 else 0.0

    out: dict = {("csd_mask", None): target.astype(np.float64), ("csd_dist", None): distance_map(target, cs)}
    for S in scales:
        shown = prompt & (labels[("weight", S)] > 0.5) if cs.prompt_weighted else prompt
        out[("csd_dip", S)] = np.where(shown, labels[("dip", S)] + noise, cs.prompt_background).astype(np.float64)
        out[("csd_prompt", S)] = shown
    return out, drawn
