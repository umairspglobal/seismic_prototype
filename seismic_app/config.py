"""Central configuration for the SAM 3 seismic segmentation app.

Keeping these values in one place matters because the noun_phrase vocabulary
must stay perfectly consistent across labeling, fine-tuning, and inference -
if a prompt string changes anywhere, SAM 3's text encoder will no longer map
it to the visual concept it was trained/prompted on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --- Tracker models (SAM 3 default, SAM 2 optional) --------------------

# Interactive point/volume tracking can run either SAM 3's tracker head
# or SAM 2.1. Text-prompt automatic detection remains SAM 3 only.
# SAM 3 is gated on the Hub: accept the license at
# https://huggingface.co/facebook/sam3 and authenticate locally
# (`hf auth login`, or set the HF_TOKEN env var) before that checkpoint
# can be downloaded. SAM 2.1 is Apache 2.0 and is not gated.

ModelFamily = Literal["sam2", "sam3"]

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
    """Normalize a UI/API family name to ``sam2`` or ``sam3``."""
    raw = (family or DEFAULT_MODEL_FAMILY).strip().lower()
    key = raw.replace(" ", "").replace("-", "").replace("_", "")
    if key in ("sam2", "sam21", "sam2.1"):
        return "sam2"
    if key in ("sam3",):
        return "sam3"
    raise ValueError(f"Unknown model family {family!r}; expected 'sam2' or 'sam3'")


def family_spec(family: str | None = None) -> dict[str, str | bool]:
    return SAM_FAMILIES[resolve_family(family)]


def checkpoint_for(family: str | None = None) -> str:
    return str(family_spec(family)["checkpoint"])

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

# --- Tiling ---------------------------------------------------------------

TILE_SIZE = 1024
TILE_STRIDE = 512  # 50% overlap

# --- Inference ------------------------------------------------------------

DETECTION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
