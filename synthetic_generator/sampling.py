"""Random draws from config ranges and smooth random fields."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi


def uniform(rng: np.random.Generator, bounds) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    return lo if hi <= lo else float(rng.uniform(lo, hi))


def integer(rng: np.random.Generator, bounds) -> int:
    lo, hi = int(bounds[0]), int(bounds[1])
    return lo if hi <= lo else int(rng.integers(lo, hi + 1))


def log_uniform(rng: np.random.Generator, bounds) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi <= lo:
        return lo
    if lo <= 0:
        return uniform(rng, bounds)
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gaussian_field(rng: np.random.Generator, shape: tuple[int, ...], sigma) -> np.ndarray:
    """Zero-mean, unit-std smooth random field (Gaussian-filtered white noise)."""
    noise = rng.standard_normal(shape)
    field = ndi.gaussian_filter(noise, sigma, mode="reflect")
    std = field.std()
    return (field - field.mean()) / std if std > 0 else field
