"""Step 5 (Mask Stitcher component): reassemble per-tile probability maps
into a full-section probability map per label, using max-fusion across
overlapping tiles.
"""

from __future__ import annotations

import numpy as np

from .preprocessing import Tile, TilingResult


def stitch_tiles(
    tile_scores: list[tuple[Tile, dict[str, np.ndarray]]],
    tiling: TilingResult,
    prompts: list[str],
) -> dict[str, np.ndarray]:
    """Combine per-tile probability maps into full-section maps.

    For pixels covered by multiple overlapping tiles, take the maximum
    predicted probability across all covering tiles (per the guide's
    reassembly step), then crop back to the section's original,
    unpadded shape.
    """
    padded_h, padded_w = tiling.padded_shape
    orig_h, orig_w = tiling.original_shape

    canvases = {p: np.zeros((padded_h, padded_w), dtype=np.float32) for p in prompts}

    for tile, scores in tile_scores:
        for prompt in prompts:
            tile_map = scores[prompt]
            th, tw = tile_map.shape
            region = canvases[prompt][tile.y : tile.y + th, tile.x : tile.x + tw]
            np.maximum(region, tile_map, out=region)

    return {p: canvases[p][:orig_h, :orig_w] for p in prompts}


def binarize(
    prob_maps: dict[str, np.ndarray], threshold: float
) -> dict[str, np.ndarray]:
    """Threshold probability maps into boolean masks."""
    return {p: (m >= threshold) for p, m in prob_maps.items()}
