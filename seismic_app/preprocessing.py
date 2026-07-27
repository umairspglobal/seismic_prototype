"""Steps 2-4 of the pipeline: normalize raw amplitudes and tile into
1024x1024 RGB patches that SAM 3's vision encoder expects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import config


def normalize_to_uint8(
    data: np.ndarray,
    low_percentile: float = config.CLIP_LOW_PERCENTILE,
    high_percentile: float = config.CLIP_HIGH_PERCENTILE,
) -> np.ndarray:
    """Percentile-clip and rescale raw amplitudes to uint8 [0, 255].

    Percentile clipping (rather than min/max) prevents a handful of
    extreme-amplitude samples from washing out the contrast of the rest
    of the section.
    """
    lo, hi = np.percentile(data, [low_percentile, high_percentile])
    if hi <= lo:
        # Degenerate/constant section - avoid divide-by-zero.
        return np.zeros_like(data, dtype=np.uint8)

    clipped = np.clip(data, lo, hi)
    normed = (clipped - lo) / (hi - lo)
    return (normed * 255).astype(np.uint8)


def to_rgb(data_uint8: np.ndarray) -> np.ndarray:
    """Channel-duplication strategy: copy the amplitude channel into R=G=B.

    Used for 2D lines, which have no neighboring sections to stack. For
    3D volumes use inline_to_rgb_25d instead.
    """
    return np.stack([data_uint8, data_uint8, data_uint8], axis=-1)


def inline_to_rgb_25d(cube_uint8: np.ndarray, iline_idx: int) -> np.ndarray:
    """True 2.5D RGB for a 3D volume: adjacent inlines in R/G/B.

    R = previous inline, G = the inline itself, B = next inline (clamped
    at the volume edges), giving SAM local 3D context for free. Input is
    the normalized uint8 cube in segyio order (n_ilines, n_xlines,
    n_samples); output is a (n_samples, n_xlines, 3) section-oriented
    image (time down), matching the 2D display convention.
    """
    n_il = cube_uint8.shape[0]
    prev_idx = max(0, iline_idx - 1)
    next_idx = min(n_il - 1, iline_idx + 1)
    # .T flips each (n_xlines, n_samples) slice to (n_samples, n_xlines).
    return np.stack(
        [cube_uint8[prev_idx].T, cube_uint8[iline_idx].T, cube_uint8[next_idx].T],
        axis=-1,
    )


@dataclass
class Tile:
    y: int
    x: int
    image: Image.Image


@dataclass
class TilingResult:
    tiles: list[Tile]
    padded_shape: tuple[int, int]  # (H, W) after padding
    original_shape: tuple[int, int]  # (H, W) before padding


def _tile_starts(dim_size: int, tile_size: int, stride: int) -> list[int]:
    """Start offsets along one axis, guaranteed to cover the whole axis.

    Seismic lines are often smaller than 1024 in one or both axes (unlike
    natural images SAM 3 was pretrained on), so a naive
    range(0, H - tile, stride) loop can silently skip data or emit zero
    tiles. This always includes a final start that reaches the edge.
    """
    if dim_size <= tile_size:
        return [0]
    starts = list(range(0, dim_size - tile_size + 1, stride))
    if starts[-1] != dim_size - tile_size:
        starts.append(dim_size - tile_size)
    return starts


def tile_image(
    rgb: np.ndarray,
    tile_size: int = config.TILE_SIZE,
    stride: int = config.TILE_STRIDE,
) -> TilingResult:
    """Pad (if needed) and split an (H, W, 3) RGB array into overlapping tiles.

    Uses 50% overlap (stride = tile_size // 2) so mask-reassembly can
    average out boundary effects near tile edges, per the guide.
    """
    h, w = rgb.shape[:2]
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)

    if pad_h or pad_w:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    padded_h, padded_w = rgb.shape[:2]

    tiles: list[Tile] = []
    for y in _tile_starts(padded_h, tile_size, stride):
        for x in _tile_starts(padded_w, tile_size, stride):
            patch = rgb[y : y + tile_size, x : x + tile_size]
            tiles.append(Tile(y=y, x=x, image=Image.fromarray(patch)))

    return TilingResult(
        tiles=tiles,
        padded_shape=(padded_h, padded_w),
        original_shape=(h, w),
    )


def sgy_to_tiles(data: np.ndarray) -> tuple[TilingResult, np.ndarray]:
    """Full preprocessing pipeline: raw amplitudes -> RGB tiles.

    Returns the TilingResult plus the full-resolution RGB uint8 image
    (useful for display/overlay later).
    """
    uint8 = normalize_to_uint8(data)
    rgb = to_rgb(uint8)
    return tile_image(rgb), rgb
