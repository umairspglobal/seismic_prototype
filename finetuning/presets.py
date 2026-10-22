"""Presets for the SAM 3 Quick Fine-Tuner.

Folder names come from the synthetic generator (``Product.folder``), using the
generator's default suffix. Each preset is one mask this trainer can learn:

* fault strokes (``Fault``)
* facies layers cut from relative geologic time (``TauAtT``)
* coarse-scale-dip horizon tracks (``CsdHorizon``)

Dip, straddle, weight, halo, the CSD dip prompt and the CSD distance map are
not text-prompt masks, so they are not presets.
"""

from __future__ import annotations

import json
from pathlib import Path

from synthetic_generator.block import FOLDER_PREFIX, Product
from synthetic_generator.config import OutputConfig
from synthetic_generator.dataset import CONFIG_NAME

DEFAULT_SUFFIX = OutputConfig().suffix


def folders_for(preset: dict, suffix: str | None = None) -> tuple[str, str]:
    """Image and label folder names for ``preset`` at ``suffix`` (default suffix if omitted)."""
    suffix = DEFAULT_SUFFIX if suffix is None else suffix
    image = Product(preset["image_kind"]).folder(suffix)
    label = Product(preset["label_kind"], preset.get("scale")).folder(suffix)
    return image, label


def dataset_suffix(root: Path) -> str:
    """Suffix of a generated dataset: ``generator_config.json``, else a ``Seis*`` folder."""
    root = Path(root)
    config = root / CONFIG_NAME
    if config.is_file():
        try:
            suffix = json.loads(config.read_text(encoding="utf-8")).get("output", {}).get("suffix")
        except (OSError, json.JSONDecodeError, AttributeError):
            suffix = None
        if isinstance(suffix, str):
            return suffix
    prefix = FOLDER_PREFIX["seis"]
    found: list[str] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.startswith(prefix):
                found.append(child.name[len(prefix):])
    if DEFAULT_SUFFIX in found or not found:
        return DEFAULT_SUFFIX
    if len(found) == 1:
        return found[0]
    return DEFAULT_SUFFIX


def _preset(*, task: str, title: str, image_kind: str, label_kind: str, mode: str, prompt: str,
            name: str, menu: str) -> tuple[str, dict]:
    spec = {
        "task": task,
        "image_kind": image_kind,
        "label_kind": label_kind,
        "scale": None,
        "mode": mode,
        "prompt": prompt,
        "name": name,
        "menu": menu,
    }
    image, label = folders_for(spec)
    spec["image"] = image
    spec["label"] = label
    if mode == "rgt_layers":
        key = f"{title}  ({image} + layers from {label})"
    else:
        key = f"{title}  ({image} + {label})"
    return key, spec


PRESETS: dict[str, dict | None] = dict((
    _preset(
        task="faults", title="Faults", image_kind="seis", label_kind="fault",
        mode="binary", prompt="fault", name="faults",
        menu="Faults model (Seis + Fault)",
    ),
    _preset(
        task="facies", title="Facies", image_kind="seis", label_kind="tau",
        mode="rgt_layers", prompt="seismic facies", name="facies",
        menu="Facies model (Seis + layers from TauAtT)",
    ),
    _preset(
        task="csd", title="CSD: coarse-scale dips", image_kind="seis", label_kind="csd_mask",
        mode="binary", prompt="horizon", name="csd",
        menu="CSD: coarse-scale dips (Seis + CsdHorizon)",
    ),
))
PRESETS["Custom"] = None
