"""Step 5/6 (Viewer + export): color-map overlay and export utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import config


def overlay_masks(
    rgb_uint8: np.ndarray,
    masks: dict[str, np.ndarray],
    alpha: float = 0.45,
    only: list[str] | None = None,
) -> Image.Image:
    """Alpha-blend colored label masks on top of the base seismic image.

    Parameters
    ----------
    rgb_uint8: (H, W, 3) uint8 base image (the full-resolution seismic
        section after channel duplication).
    masks: noun_phrase -> boolean array (H, W).
    only: optional subset of noun_phrases to draw (for layer toggling in
        the viewer); defaults to all labels in masks.
    """
    base = Image.fromarray(rgb_uint8).convert("RGBA")
    composite = base

    labels = only if only is not None else list(masks.keys())
    for label in labels:
        mask = masks.get(label)
        if mask is None or not mask.any():
            continue
        color = config.LABEL_COLORS.get(label, (255, 255, 255))
        overlay = Image.new("RGBA", base.size, color + (0,))
        alpha_channel = (mask.astype(np.uint8) * int(255 * alpha))
        overlay.putalpha(Image.fromarray(alpha_channel, mode="L"))
        composite = Image.alpha_composite(composite, overlay)

    return composite


def save_overlay_png(image: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def save_masks_npz(masks: dict[str, np.ndarray], path: str | Path) -> None:
    """Export per-label boolean masks as a single compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # npz keys can't contain spaces safely across all tools; keep them but
    # np.savez supports arbitrary string keys via **kwargs is not possible
    # with spaces, so pass a dict directly.
    np.savez_compressed(path, **{k.replace(" ", "_"): v for k, v in masks.items()})
