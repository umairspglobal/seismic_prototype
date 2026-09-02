"""Step 9 of the SAM 3 fine-tuning workflow: standalone per-image and mean
IoU calculation for the "seismic facies" checkpoint on the held-out val
split, run entirely through this app's own inference path
(seismic_app.inference.Sam3SeismicSegmenter) so the number you get here is
exactly what you'd see from the app/CLI with the same checkpoint - no
separate eval-only code path to keep in sync.

Usage:
    python -m finetuning.validate_checkpoint \
        --dataset-root finetune_data \
        --checkpoint /path/to/converted/hf/checkpoint_dir

Run this BEFORE and AFTER fine-tuning (pointing --checkpoint at
facebook/sam3 for the "before" baseline) to see the mIoU improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from seismic_app import config
from seismic_app.inference import Sam3SeismicSegmenter

NOUN_PHRASE = "seismic facies"


def _ground_truth_masks(coco: dict, images_dir: Path) -> dict[int, np.ndarray]:
    """image_id -> (H, W) bool mask, zeros for images with no annotation."""
    by_image: dict[int, np.ndarray] = {}
    images_by_id = {img["id"]: img for img in coco["images"]}
    for img in coco["images"]:
        by_image[img["id"]] = np.zeros((img["height"], img["width"]), dtype=bool)
    for ann in coco["annotations"]:
        decoded = mask_utils.decode(ann["segmentation"]).astype(bool)
        by_image[ann["image_id"]] |= decoded
    return by_image


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0  # both empty - perfect agreement
    return float(intersection / union)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Output dir from prepare_dataset.py")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--checkpoint", default=config.DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=config.MASK_THRESHOLD)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N images")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    ann_path = root / f"{args.split}.json"
    images_dir = root / "images" / args.split
    coco = json.loads(ann_path.read_text())
    gt_masks = _ground_truth_masks(coco, images_dir)

    images = coco["images"]
    if args.limit:
        images = images[: args.limit]

    print(f"Loading checkpoint '{args.checkpoint}'...")
    segmenter = Sam3SeismicSegmenter(
        checkpoint=args.checkpoint, device=args.device, prompts=[NOUN_PHRASE]
    )

    ious: list[float] = []
    for img in images:
        pil_image = Image.open(images_dir / img["file_name"]).convert("RGB")
        probs = segmenter.segment_tile(pil_image)[NOUN_PHRASE]
        pred_mask = probs > args.threshold
        gt_mask = gt_masks[img["id"]]
        ious.append(_iou(pred_mask, gt_mask))

    ious_arr = np.array(ious)
    print(f"\n{args.split}: {len(ious_arr)} images evaluated")
    print(f"mean IoU ('{NOUN_PHRASE}'): {ious_arr.mean():.4f}")
    print(f"median IoU: {np.median(ious_arr):.4f}")
    print(f"images with IoU == 0.00: {(ious_arr == 0).sum()} "
          f"({(ious_arr == 0).mean():.1%}) - if this is most of the val set, "
          "see the 'stuck val mIoU at 0.00' entry in finetuning/README.md")


if __name__ == "__main__":
    main()
