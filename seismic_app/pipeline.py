"""Top-level orchestration: .sgy -> preprocess -> tile -> SAM 3 infer ->
stitch -> export. This is the "no prompts at inference time" entry point:
callers just supply a file path and get all five feature masks back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .inference import Sam3SeismicSegmenter
from .preprocessing import sgy_to_tiles
from .sgy_loader import load_2d_section
from .stitching import binarize, stitch_tiles
from .visualization import overlay_masks, save_masks_npz, save_overlay_png


@dataclass
class SegmentationResult:
    rgb: np.ndarray  # (H, W, 3) uint8 base image
    prob_maps: dict[str, np.ndarray]  # noun_phrase -> float32 (H, W)
    masks: dict[str, np.ndarray]  # noun_phrase -> bool (H, W)


def run_on_file(
    path: str | Path,
    segmenter: Sam3SeismicSegmenter,
    prompts: list[str] | None = None,
    threshold: float = config.MASK_THRESHOLD,
) -> SegmentationResult:
    """Run the full pipeline on a single .sgy file, no user prompts needed."""
    prompts = prompts or config.SEISMIC_PROMPTS

    data = load_2d_section(path)
    tiling, rgb = sgy_to_tiles(data)

    tile_scores = []
    for tile in tiling.tiles:
        scores = segmenter.segment_tile(tile.image)
        tile_scores.append((tile, scores))

    prob_maps = stitch_tiles(tile_scores, tiling, prompts)
    masks = binarize(prob_maps, threshold)

    return SegmentationResult(rgb=rgb, prob_maps=prob_maps, masks=masks)


def process_and_export(
    path: str | Path,
    out_dir: str | Path,
    segmenter: Sam3SeismicSegmenter | None = None,
    checkpoint: str = config.DEFAULT_CHECKPOINT,
    device: str | None = None,
    threshold: float = config.MASK_THRESHOLD,
) -> SegmentationResult:
    """Convenience wrapper: run the pipeline and write PNG + NPZ outputs."""
    if segmenter is None:
        segmenter = Sam3SeismicSegmenter(checkpoint=checkpoint, device=device)

    result = run_on_file(path, segmenter, threshold=threshold)

    out_dir = Path(out_dir)
    stem = Path(path).stem

    overlay = overlay_masks(result.rgb, result.masks)
    save_overlay_png(overlay, out_dir / f"{stem}_overlay.png")
    save_masks_npz(result.masks, out_dir / f"{stem}_masks.npz")

    return result
