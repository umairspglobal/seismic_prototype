"""Supervision arrays for one section: dip, straddle, scale weight and halo.

Rules (reverse-engineered from training_10_7_26 and the generator description):

dip       ``tau[y, x] - tau[y, x + S]`` - the vertical shift (in tau samples) to the
          same reflector ``S`` traces to the right; columns without a partner
          (``x >= W - S``) repeat the last valid column, then the field is lightly
          Gaussian-smoothed.
straddle  1 where ``x`` and ``x + S`` are both off the fault stroke, ``x + S`` is
          inside the section and a fault trace crosses the row strictly between
          them (rows past a fault tip, i.e. its rounded end cap, do not count).
weight    1 except 0 for ``x >= W - S - 1``; in a corridor around every stroke
          (``corridor_pad`` px either side); where the partner ``x + S`` falls in
          the stroke widened by ``partner_pad_left`` / ``partner_pad_right`` (with
          S = 12 the two corridors leave the one-pixel notch 3-4 columns before the
          stroke); where the partner reflector leaves the section (its tau is
          outside the range of column ``x + S``); and, optionally, where ``x`` and
          ``x + S`` lie in different packages (the pair crosses an unconformity, so
          the tau difference contains the hiatus instead of a reflector shift).
halo      Gaussian blur of the stroke with radius ``S`` (sigma = factor * S), scaled
          by its ``halo_quantile`` value and clipped, so it peaks at 1.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .config import LabelConfig


def window_any(mask: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """``out[y, x] = any(mask[y, x + lo : x + hi + 1])`` with the window clipped to the row."""
    H, W = mask.shape
    csum = np.zeros((H, W + 1), dtype=np.int32)
    np.cumsum(mask, axis=1, out=csum[:, 1:])
    xs = np.arange(W)
    a = np.clip(xs + lo, 0, W)
    b = np.clip(xs + hi + 1, 0, W)
    return (csum[:, b] - csum[:, a]) > 0


def dip(tau: np.ndarray, S: int, smoothing: float = 1.0, fill: str = "edge") -> np.ndarray:
    H, W = tau.shape
    out = np.empty_like(tau, dtype=np.float64)
    out[:, : W - S] = tau[:, : W - S] - tau[:, S:]
    out[:, W - S:] = out[:, W - S - 1: W - S] if fill == "edge" else 0.0
    if smoothing > 0:
        out = ndi.gaussian_filter(out, smoothing, mode="nearest")
    return out


def straddle(mask: np.ndarray, offsets: list[np.ndarray], inside: list[np.ndarray], S: int) -> np.ndarray:
    H, W = mask.shape
    out = np.zeros((H, W), dtype=bool)
    if S >= W:
        return out
    a, b = slice(0, W - S), slice(S, W)
    cross = np.zeros((H, W - S), dtype=bool)
    for dx, ins in zip(offsets, inside):
        cross |= ins[:, a] & ins[:, b] & (dx[:, a] < 0) & (dx[:, b] > 0)
    out[:, : W - S] = cross & ~mask[:, a] & ~mask[:, b]
    return out


def weight(mask: np.ndarray, tau: np.ndarray, S: int, cfg: LabelConfig, package: np.ndarray | None = None) -> np.ndarray:
    H, W = mask.shape
    out = np.ones((H, W), dtype=np.float64)
    out[:, max(0, W - S - 1):] = 0.0
    if mask.any():
        out[window_any(mask, -cfg.corridor_pad, cfg.corridor_pad)] = 0.0
        partner = window_any(mask, -cfg.partner_pad_right, cfg.partner_pad_left)
        out[:, : W - S][partner[:, S:]] = 0.0
    if cfg.partner_outside and S < W:
        lo, hi = tau.min(axis=0), tau.max(axis=0)
        here = tau[:, : W - S]
        gone = (here < lo[None, S:]) | (here > hi[None, S:])
        out[:, : W - S][gone] = 0.0
    if cfg.unconformity_rule and package is not None and S < W and package.max() > 0:
        out[:, : W - S][package[:, : W - S] != package[:, S:]] = 0.0
    return out


def halo(mask: np.ndarray, S: int, cfg: LabelConfig) -> np.ndarray:
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float64)
    sigma = max(cfg.halo_sigma_factor * S, 1e-3)
    truncate = max(S / sigma, 0.5)
    m = mask.astype(np.float64)
    if cfg.halo_mode == "horizontal":
        blur = ndi.gaussian_filter1d(m, sigma, axis=1, mode="reflect", truncate=truncate)
    else:
        blur = ndi.gaussian_filter(m, sigma, mode="reflect", truncate=truncate)
    ref = float(np.quantile(blur, cfg.halo_quantile))
    if ref <= 1e-12:
        ref = float(blur.max())
    return np.clip(blur / ref, 0.0, 1.0)


def section_labels(tau: np.ndarray, mask: np.ndarray, offsets: list[np.ndarray], inside: list[np.ndarray],
                   scales: list[int], cfg: LabelConfig, package: np.ndarray | None = None
                   ) -> dict[tuple[str, int], np.ndarray]:
    out: dict[tuple[str, int], np.ndarray] = {}
    for S in scales:
        out[("dip", S)] = dip(tau, S, cfg.dip_smoothing, cfg.dip_fill)
        out[("weight", S)] = weight(mask, tau, S, cfg, package)
        out[("straddle", S)] = straddle(mask, offsets, inside, S).astype(np.float64)
        out[("halo", S)] = halo(mask, S, cfg)
    return out
