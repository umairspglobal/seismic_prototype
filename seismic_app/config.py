"""Central configuration for the SAM 3 seismic segmentation app.

Keeping these values in one place matters because the noun_phrase vocabulary
must stay perfectly consistent across labeling, fine-tuning, and inference -
if a prompt string changes anywhere, SAM 3's text encoder will no longer map
it to the visual concept it was trained/prompted on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# --- Tracker models (SAM 3 default, SAM 3.1 / SAM 2 optional) ----------

# Interactive point/volume tracking can run either SAM 3's tracker head
# through transformers, SAM 3.1 Object Multiplex through Meta's native
# repository, or SAM 2.1. Text-prompt automatic detection remains SAM 3 only.
# SAM 3 is gated on the Hub: accept the license at
# https://huggingface.co/facebook/sam3 and authenticate locally
# (`hf auth login`, or set the HF_TOKEN env var) before that checkpoint
# can be downloaded. SAM 2.1 is Apache 2.0 and is not gated.

ModelFamily = Literal["sam2", "sam3", "sam31"]

DEFAULT_MODEL_FAMILY: ModelFamily = "sam3"

SAM_FAMILIES: dict[str, dict[str, str | bool]] = {
    "sam3": {
        "id": "sam3",
        "label": "SAM 3",
        "checkpoint": "facebook/sam3",
        "architecture": "SAM 3 (Hugging Face transformers)",
        "point_model": "Sam3TrackerModel",
        "video_model": "Sam3TrackerVideoModel",
        "gated": True,
    },
    "sam31": {
        "id": "sam31",
        "label": "SAM 3.1",
        "checkpoint": "facebook/sam3.1",
        "architecture": "SAM 3.1 Object Multiplex (facebookresearch/sam3)",
        "point_model": "Sam3MultiplexVideoPredictor",
        "video_model": "Sam3MultiplexVideoPredictor",
        "gated": True,
    },
    "sam2": {
        "id": "sam2",
        "label": "SAM 2",
        "checkpoint": "facebook/sam2.1-hiera-large",
        "architecture": "SAM 2.1 (Hugging Face transformers)",
        "point_model": "Sam2Model",
        "video_model": "Sam2VideoModel",
        "gated": False,
    },
}

DEFAULT_CHECKPOINT = str(SAM_FAMILIES[DEFAULT_MODEL_FAMILY]["checkpoint"])


def resolve_family(family: str | None = None) -> ModelFamily:
    """Normalize a UI/API family name to a supported tracker family."""
    raw = (family or DEFAULT_MODEL_FAMILY).strip().lower()
    key = raw.replace(" ", "").replace("-", "").replace("_", "")
    if key in ("sam2", "sam21", "sam2.1"):
        return "sam2"
    if key in ("sam3",):
        return "sam3"
    if key in ("sam31", "sam3.1"):
        return "sam31"
    raise ValueError(
        f"Unknown model family {family!r}; expected 'sam2', 'sam3', or 'sam31'"
    )


def family_spec(family: str | None = None) -> dict[str, str | bool]:
    return SAM_FAMILIES[resolve_family(family)]


def checkpoint_for(family: str | None = None) -> str:
    return str(family_spec(family)["checkpoint"])


# --- Text-prompt detector (fine-tuned "seismic facies") ----------------
# Distinct from SAM_FAMILIES["sam3"]["checkpoint"], which the React
# click/volume tracker loads. Phase 1 fine-tuning produces a Sam3Model
# directory, not a tracker checkpoint. convert_sam3_to_hf.py writes those
# folders under finetuned_checkpoints/; the React dropdown lists them.

_REPO_ROOT = Path(__file__).resolve().parents[1]
FINETUNED_CHECKPOINTS_DIR = _REPO_ROOT / "finetuned_checkpoints"
DEFAULT_TEXT_CHECKPOINT_DIR = FINETUNED_CHECKPOINTS_DIR / "seismic_facies_phase1"
OFFICIAL_TEXT_CHECKPOINT = "facebook/sam3"
FACIES_PROMPT = "seismic facies"

# Runtime override from POST /api/text-checkpoint. None means "use env /
# first converted folder / official SAM 3" as text_checkpoint() describes.
_TEXT_CHECKPOINT_OVERRIDE: str | None = None


def _is_hf_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _checkpoint_key(path: str) -> str:
    """Compare hub ids and local folders without slash/case noise."""
    raw = path.strip().replace("\\", "/")
    local = Path(path.strip())
    try:
        if local.exists():
            return str(local.resolve()).lower()
    except OSError:
        pass
    return raw.lower()


def list_text_checkpoints() -> list[dict[str, str]]:
    """Official SAM 3 plus every converted HF folder under finetuned_checkpoints/."""
    items: list[dict[str, str]] = [
        {
            "id": OFFICIAL_TEXT_CHECKPOINT,
            "label": "SAM 3 (official)",
            "path": OFFICIAL_TEXT_CHECKPOINT,
            "source": "official",
        }
    ]
    seen = {_checkpoint_key(OFFICIAL_TEXT_CHECKPOINT)}
    if FINETUNED_CHECKPOINTS_DIR.is_dir():
        for child in sorted(FINETUNED_CHECKPOINTS_DIR.iterdir(), key=lambda p: p.name.lower()):
            if not _is_hf_model_dir(child):
                continue
            key = _checkpoint_key(str(child))
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "id": child.name,
                    "label": child.name,
                    "path": str(child),
                    "source": "local",
                }
            )
    env = os.environ.get("SAM3_TEXT_CHECKPOINT", "").strip()
    if env and _checkpoint_key(env) not in seen:
        items.append(
            {
                "id": Path(env).name or env,
                "label": Path(env).name or env,
                "path": env,
                "source": "env",
            }
        )
    return items


def resolve_text_checkpoint(path: str) -> str:
    """Accept a hub id, an HF folder, or a name under finetuned_checkpoints/."""
    raw = (path or "").strip()
    if not raw:
        raise ValueError("checkpoint path is empty")
    if raw in (OFFICIAL_TEXT_CHECKPOINT, "sam3", "SAM 3"):
        return OFFICIAL_TEXT_CHECKPOINT
    candidate = Path(raw)
    if _is_hf_model_dir(candidate):
        return str(candidate)
    nested = FINETUNED_CHECKPOINTS_DIR / raw
    if _is_hf_model_dir(nested):
        return str(nested)
    raise ValueError(
        f"No converted SAM 3 checkpoint at {raw!r}; "
        "need a Hugging Face folder with config.json "
        f"(convert into {FINETUNED_CHECKPOINTS_DIR})."
    )


def set_text_checkpoint(path: str) -> str:
    """Pin the detector to ``path`` until the process exits or it is set again."""
    global _TEXT_CHECKPOINT_OVERRIDE
    _TEXT_CHECKPOINT_OVERRIDE = resolve_text_checkpoint(path)
    return _TEXT_CHECKPOINT_OVERRIDE


def text_checkpoint() -> str:
    """HF dir or hub id for Sam3SeismicSegmenter.

    Preference: UI/API override, then ``SAM3_TEXT_CHECKPOINT``, then the
    default converted folder when ``config.json`` is present, else stock
    SAM 3.
    """
    if _TEXT_CHECKPOINT_OVERRIDE:
        return _TEXT_CHECKPOINT_OVERRIDE
    env = os.environ.get("SAM3_TEXT_CHECKPOINT", "").strip()
    if env:
        return env
    if _is_hf_model_dir(DEFAULT_TEXT_CHECKPOINT_DIR):
        return str(DEFAULT_TEXT_CHECKPOINT_DIR)
    return DEFAULT_CHECKPOINT

# --- Fixed noun_phrase vocabulary --------------------------------------
# Order matters only for the color map below; SAM 3 is prompted with each
# phrase independently.

SEISMIC_PROMPTS: list[str] = [
    "fault",
    "channel",
    "seismic facies",
    "salt body",
    "horizon",
]


@dataclass(frozen=True)
class LabelStyle:
    noun_phrase: str
    color: tuple[int, int, int]  # RGB, 0-255


# Visualization colors per the guide: faults=red, channels=blue,
# facies=green, salt=yellow, horizons=cyan.
LABEL_STYLES: list[LabelStyle] = [
    LabelStyle("fault", (255, 0, 0)),
    LabelStyle("channel", (0, 0, 255)),
    LabelStyle("seismic facies", (0, 200, 0)),
    LabelStyle("salt body", (255, 220, 0)),
    LabelStyle("horizon", (0, 255, 255)),
]

LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    style.noun_phrase: style.color for style in LABEL_STYLES
}

# --- Preprocessing ------------------------------------------------------

CLIP_LOW_PERCENTILE = 2.0
CLIP_HIGH_PERCENTILE = 98.0

# --- Large files ----------------------------------------------------------
# A file takes the disk-cache path when loading it into RAM would need
# more than this fraction of physical memory. Loading costs about
# IN_MEMORY_BYTES_PER_VALUE bytes per sample: the float32 cube plus the
# clip/scale temporaries of normalize_to_uint8 and the uint8 copy.
IN_MEMORY_BUDGET_FRACTION = float(os.environ.get("SEISMIC_IN_MEMORY_FRACTION", "0.5"))
IN_MEMORY_BYTES_PER_VALUE = 18
# Absolute override in GB (e.g. to force the cache path in tests).
IN_MEMORY_BUDGET_GB = os.environ.get("SEISMIC_IN_MEMORY_BUDGET_GB")

CACHE_DIR = Path(os.environ.get("SEISMIC_CACHE_DIR", str(_REPO_ROOT / "outputs" / "cache")))
# Traces sampled to estimate the clip percentiles of a large file.
CACHE_PERCENTILE_TRACES = 20_000
# Float32 bytes per conversion block.
CACHE_BLOCK_BYTES = 256 * 1024**2

# 2D lines wider than this are served as overlapping pages of traces.
MAX_2D_PAGE_TRACES = 4096
PAGE_OVERLAP_TRACES = 256

# Share of physical RAM a single propagation may use for its frames,
# processor tensors and masks when no explicit window is requested.
PROPAGATION_RAM_FRACTION = float(os.environ.get("SEISMIC_PROPAGATION_RAM_FRACTION", "0.25"))
# SAM video processors resize every frame to this square before encoding.
PROPAGATION_MODEL_SIDE = 1008
PROPAGATION_MIN_FRAMES = 8

# --- Tiling ---------------------------------------------------------------

TILE_SIZE = 1024
TILE_STRIDE = 512  # 50% overlap

# --- Inference ------------------------------------------------------------

DETECTION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
