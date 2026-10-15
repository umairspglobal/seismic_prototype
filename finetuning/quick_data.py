"""Data loading for the quick SAM 3 fine-tuner (``finetuning.quick_finetune``).

The training folders hold one 2D section per file, matched across folders by
file name (``Seis75/17.dat`` <-> ``Fault75/17.dat``). Each ``.dat`` file is a
raw little-endian float64 square image (224 x 224 for training_10_7_26),
time/depth down and traces across - the same orientation the app displays.

Label modes
-----------
``binary``     the label file already is a 0/1 mask (Fault75).
``rgt_layers`` the label file is relative geologic time (TauAtT75). It is
               normalized per section, cut into ``num_layers`` equal bands
               and every second band (counting from the top, top band off)
               becomes foreground, so the mask outlines stratigraphic
               packages and their offsets across faults.
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from seismic_app.preprocessing import normalize_to_uint8

LABEL_MODES = ("binary", "rgt_layers")


def _numeric_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), stem) if stem.isdigit() else (1 << 62, stem)


def list_pairs(image_dir: Path, label_dir: Path) -> list[str]:
    """Stems present in both folders (``.dat`` or ``.npy``), numerically sorted."""
    labels = {p.stem for p in label_dir.iterdir() if p.suffix.lower() in (".dat", ".npy")}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in (".dat", ".npy")]
    return [p.stem for p in sorted(images, key=_numeric_key) if p.stem in labels]


def _resolve(folder: Path, stem: str) -> Path:
    for suffix in (".dat", ".npy"):
        path = folder / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"{folder}/{stem}.dat")


def read_section(path: Path) -> np.ndarray:
    """Read one square 2D section as float32 (raw float64/float32 or .npy)."""
    if path.suffix.lower() == ".npy":
        data = np.load(path)
        if data.ndim != 2:
            raise ValueError(f"{path}: expected a 2D array, got shape {data.shape}")
        return data.astype(np.float32)
    size = path.stat().st_size
    for dtype, width in (("<f8", 8), ("<f4", 4)):
        if size % width:
            continue
        side = math.isqrt(size // width)
        if side * side * width == size:
            return np.fromfile(path, dtype=dtype).reshape(side, side).astype(np.float32)
    raise ValueError(f"{path}: {size} bytes is not a square float64/float32 image")


def rgt_to_layers(rgt: np.ndarray, num_layers: int) -> np.ndarray:
    lo, hi = float(rgt.min()), float(rgt.max())
    if hi <= lo:
        return np.zeros(rgt.shape, dtype=bool)
    band = np.floor((rgt - lo) / (hi - lo) * num_layers).astype(np.int32)
    band = np.clip(band, 0, num_layers - 1)
    return (band % 2) == 1


def make_label(raw: np.ndarray, mode: str, num_layers: int = 8) -> np.ndarray:
    if mode == "binary":
        return raw > 0.5
    if mode == "rgt_layers":
        return rgt_to_layers(raw, num_layers)
    raise ValueError(f"Unknown label mode {mode!r}; expected one of {LABEL_MODES}")


@dataclass
class SectionSet:
    """Preloaded uint8 sections and boolean masks, stacked (N, H, W)."""

    stems: list[str]
    images: np.ndarray
    masks: np.ndarray

    def __len__(self) -> int:
        return len(self.stems)

    @property
    def positive_fraction(self) -> float:
        return float(self.masks.mean()) if len(self) else 0.0


def load_sections(
    image_dir: Path,
    label_dir: Path,
    stems: list[str],
    label_mode: str,
    num_layers: int = 8,
    progress: Callable[[int, int], None] | None = None,
    workers: int = 8,
) -> SectionSet:
    """Read, normalize (same percentile clip as the app) and label ``stems``."""

    def load_one(stem: str) -> tuple[np.ndarray, np.ndarray]:
        seis = read_section(_resolve(image_dir, stem))
        label = read_section(_resolve(label_dir, stem))
        if seis.shape != label.shape:
            raise ValueError(f"{stem}: image {seis.shape} and label {label.shape} differ")
        return normalize_to_uint8(seis), make_label(label, label_mode, num_layers)

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (img, mask) in enumerate(pool.map(load_one, stems), start=1):
            images.append(img)
            masks.append(mask)
            if progress and (i % 100 == 0 or i == len(stems)):
                progress(i, len(stems))
    return SectionSet(list(stems), np.stack(images), np.stack(masks))


def split_stems(
    stems: list[str],
    max_samples: int,
    val_fraction: float,
    seed: int,
    max_val: int = 0,
) -> tuple[list[str], list[str]]:
    """Shuffle, keep ``max_samples`` (0 = all), then hold out a validation share."""
    rng = random.Random(seed)
    chosen = stems[:]
    rng.shuffle(chosen)
    if max_samples > 0:
        chosen = chosen[:max_samples]
    if len(chosen) < 2:
        raise ValueError("Need at least 2 matched samples (1 train + 1 val).")
    n_val = max(1, round(len(chosen) * val_fraction))
    if max_val > 0:
        n_val = min(n_val, max_val)
    n_val = min(n_val, len(chosen) - 1)
    return chosen[n_val:], chosen[:n_val]


def build_mosaic(
    sections: SectionSet,
    indices: list[int],
    grid: int,
    flips: list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tile ``grid x grid`` sections into one (grid*H, grid*W) image + mask.

    Mosaics let the model see seismic at roughly the pixel scale of the
    app's 1024-px tiles instead of one 224-px section stretched 4.5x.
    """
    if not indices:
        raise ValueError("A mosaic needs at least one section")
    _n, h, w = sections.images.shape
    image = np.empty((grid * h, grid * w), dtype=np.uint8)
    mask = np.empty((grid * h, grid * w), dtype=bool)
    # Repeat sections when a batch runs short, so no cell stays uninitialized.
    for k in range(grid * grid):
        r, c = divmod(k, grid)
        img, msk = sections.images[indices[k % len(indices)]], sections.masks[indices[k % len(indices)]]
        if flips and flips[k % len(flips)]:
            img, msk = img[:, ::-1], msk[:, ::-1]
        image[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
        mask[r * h : (r + 1) * h, c * w : (c + 1) * w] = msk
    return image, mask


def fixed_mosaics(sections: SectionSet, grid: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic, non-repeating mosaics covering every section once (validation)."""
    per = grid * grid
    order = list(range(len(sections)))
    out = []
    for start in range(0, len(order), per):
        chunk = order[start : start + per]
        while len(chunk) < per:
            chunk.append(order[(start + len(chunk)) % len(order)])
        out.append(build_mosaic(sections, chunk, grid))
    return out
