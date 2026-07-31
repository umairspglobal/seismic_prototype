"""Top-level orchestration: .sgy -> geometry + preprocess -> tile ->
SAM 3 infer -> stitch -> geometry-aware export.

Geometry is extracted once at load time and threaded through every stage
so the exported masks land exactly where the seismic sits in ParaView.
Array orientation everywhere: 2D sections and per-inline slices are
(n_samples, n_traces) - time down, traces across.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .geometry import SectionGeometry
from .inference import Sam3SeismicSegmenter
from .logutil import get_logger
from .preprocessing import inline_to_rgb_25d, normalize_to_uint8, tile_image, to_rgb
from .sgy_loader import load_any
from .stitching import binarize, stitch_tiles
from .visualization import overlay_masks, save_masks_npz, save_overlay_png
from .vtk_export import export_masks

log = get_logger("pipeline")


@dataclass
class SegmentationResult:
    """Masks plus the geometry needed to place them in the real world.

    For 2D lines (geometry.kind == "2d"), every array is a
    (n_samples, n_traces) section. For 3D volumes ("3d"), masks and
    prob_maps are (n_ilines, n_samples, n_xlines) stacks - one section
    per inline - and rgb/amplitude hold the middle inline for preview.
    """

    geometry: SectionGeometry
    rgb: np.ndarray  # (H, W, 3) uint8 preview image
    amplitude: np.ndarray  # float32 raw amplitudes (section or cube)
    prob_maps: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]


def _segment_section_rgb(
    rgb: np.ndarray,
    segmenter: Sam3SeismicSegmenter,
    prompts: list[str],
) -> dict[str, np.ndarray]:
    """Tile one (H, W, 3) image, run SAM 3, stitch back to full size."""
    tiling = tile_image(rgb)
    n_tiles = len(tiling.tiles)
    log.info(
        "Tiled section %dx%d into %d tiles; running text prompts on each...",
        rgb.shape[1],
        rgb.shape[0],
        n_tiles,
    )
    tile_scores = []
    for i, tile in enumerate(tiling.tiles, start=1):
        t0 = time.perf_counter()
        log.info("  tile %d/%d at (y=%d, x=%d)...", i, n_tiles, tile.y, tile.x)
        scores = segmenter.segment_tile(tile.image)
        log.info("  tile %d/%d done in %.1fs", i, n_tiles, time.perf_counter() - t0)
        tile_scores.append((tile, scores))
    return stitch_tiles(tile_scores, tiling, prompts)


def run_on_section(
    section: np.ndarray,
    geometry: SectionGeometry,
    segmenter: Sam3SeismicSegmenter,
    prompts: list[str] | None = None,
    threshold: float = config.MASK_THRESHOLD,
) -> SegmentationResult:
    """Run the text-prompt pipeline on a loaded (n_samples, n_traces) section."""
    prompts = prompts or config.SEISMIC_PROMPTS
    log.info(
        "Text-prompt segmentation on 2D section shape=%s (time x traces), threshold=%.2f",
        section.shape,
        threshold,
    )
    t0 = time.perf_counter()

    rgb = to_rgb(normalize_to_uint8(section))
    prob_maps = _segment_section_rgb(rgb, segmenter, prompts)
    masks = binarize(prob_maps, threshold)

    for name, mask in masks.items():
        log.info("  mask '%s': %.2f%% coverage", name, 100.0 * float(mask.mean()))
    log.info("Section segmentation finished in %.1fs", time.perf_counter() - t0)

    return SegmentationResult(
        geometry=geometry,
        rgb=rgb,
        amplitude=section,
        prob_maps=prob_maps,
        masks=masks,
    )


def run_on_volume(
    cube: np.ndarray,
    geometry: SectionGeometry,
    segmenter: Sam3SeismicSegmenter,
    prompts: list[str] | None = None,
    threshold: float = config.MASK_THRESHOLD,
) -> SegmentationResult:
    """Run the pipeline inline-by-inline over a 3D cube with 2.5D RGB.

    Each inline is presented to SAM as an RGB image whose channels are
    the previous/current/next inlines, so the model sees local 3D
    context. Masks are stacked to (n_ilines, n_samples, n_xlines).
    """
    prompts = prompts or config.SEISMIC_PROMPTS

    cube_u8 = normalize_to_uint8(cube)
    n_il = cube.shape[0]
    log.info(
        "Text-prompt segmentation on 3D volume shape=%s (%d inlines), threshold=%.2f",
        cube.shape,
        n_il,
        threshold,
    )
    t0 = time.perf_counter()

    prob_stacks: dict[str, list[np.ndarray]] = {p: [] for p in prompts}
    preview_rgb: np.ndarray | None = None

    for il in range(n_il):
        log.info("Inline %d/%d...", il + 1, n_il)
        rgb = inline_to_rgb_25d(cube_u8, il)
        if il == n_il // 2:
            preview_rgb = rgb
        prob_maps = _segment_section_rgb(rgb, segmenter, prompts)
        for p in prompts:
            prob_stacks[p].append(prob_maps[p])

    prob_maps_3d = {p: np.stack(prob_stacks[p], axis=0) for p in prompts}
    masks_3d = binarize(prob_maps_3d, threshold)
    log.info("Volume segmentation finished in %.1fs", time.perf_counter() - t0)

    return SegmentationResult(
        geometry=geometry,
        rgb=preview_rgb,
        amplitude=cube,
        prob_maps=prob_maps_3d,
        masks=masks_3d,
    )


def run_on_file(
    path: str | Path,
    segmenter: Sam3SeismicSegmenter,
    prompts: list[str] | None = None,
    threshold: float = config.MASK_THRESHOLD,
) -> SegmentationResult:
    """Load a .sgy file (auto-detecting 2D vs 3D) and segment it."""
    data, geometry = load_any(path)
    if geometry.kind == "3d":
        return run_on_volume(data, geometry, segmenter, prompts, threshold)
    return run_on_section(data, geometry, segmenter, prompts, threshold)


def process_and_export(
    path: str | Path,
    out_dir: str | Path,
    segmenter: Sam3SeismicSegmenter | None = None,
    checkpoint: str = config.DEFAULT_CHECKPOINT,
    device: str | None = None,
    threshold: float = config.MASK_THRESHOLD,
    trace_spacing: float | None = None,
    sample_interval: float | None = None,
) -> SegmentationResult:
    """Run the pipeline and write PNG + NPZ + VTK outputs.

    trace_spacing (m) and sample_interval (ms) are optional *overrides*;
    by default both are read from the SEG-Y headers.
    """
    if segmenter is None:
        segmenter = Sam3SeismicSegmenter(checkpoint=checkpoint, device=device)

    result = run_on_file(path, segmenter, threshold=threshold)

    geometry = result.geometry
    if trace_spacing is not None:
        geometry.trace_spacing_m = trace_spacing
    if sample_interval is not None:
        geometry.dt_ms = sample_interval

    out_dir = Path(out_dir)
    stem = Path(path).stem

    if geometry.kind == "2d":
        overlay = overlay_masks(result.rgb, result.masks)
        save_overlay_png(overlay, out_dir / f"{stem}_overlay.png")
    else:
        # 3D: preview overlay of the middle inline.
        mid = result.amplitude.shape[0] // 2
        mid_masks = {p: m[mid] for p, m in result.masks.items()}
        overlay = overlay_masks(result.rgb, mid_masks)
        save_overlay_png(overlay, out_dir / f"{stem}_overlay_il{mid}.png")

    save_masks_npz(result.masks, out_dir / f"{stem}_masks.npz")
    export_masks(result.masks, result.amplitude, geometry, out_dir / f"{stem}_masks")

    return result
