"""Central configuration for the SAM 3 seismic segmentation app.

Keeping these values in one place matters because the noun_phrase vocabulary
must stay perfectly consistent across labeling, fine-tuning, and inference -
if a prompt string changes anywhere, SAM 3's text encoder will no longer map
it to the visual concept it was trained/prompted on.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- SAM 3 model -------------------------------------------------------

# Gated on the Hub. The user must accept the license at
# https://huggingface.co/facebook/sam3 and authenticate locally
# (`hf auth login`, or set the HF_TOKEN env var) before this checkpoint
# can be downloaded.
DEFAULT_CHECKPOINT = "facebook/sam3"

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
