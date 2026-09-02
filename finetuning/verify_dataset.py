"""Step 2 of the SAM 3 fine-tuning workflow: sanity-check a COCO JSON before
spending GPU time training on it.

Checks performed:
- every image file referenced in the JSON actually exists and its size
  matches the recorded height/width;
- every annotation's noun_phrase matches its category's noun_phrase (a
  drifted noun_phrase silently breaks SAM 3's text-conditioned training,
  since it looks up masks by the same phrase used at inference time);
- annotation counts per class, and positive/negative image ratio, so
  class imbalance is visible before training rather than discovered from
  a stuck val/mIoU curve later.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from pycocotools import mask as mask_utils


def verify_split(ann_path: Path, images_dir: Path) -> bool:
    coco = json.loads(ann_path.read_text())
    categories = {c["id"]: c for c in coco["categories"]}
    images_by_id = {img["id"]: img for img in coco["images"]}

    ok = True
    print(f"\n=== {ann_path.name} ===")
    print(f"images: {len(coco['images'])}, annotations: {len(coco['annotations'])}")

    # 1. Image files exist and dimensions match.
    missing = 0
    mismatched_size = 0
    for img in coco["images"]:
        path = images_dir / img["file_name"]
        if not path.exists():
            missing += 1
            continue
        with Image.open(path) as im:
            if im.size != (img["width"], img["height"]):
                mismatched_size += 1
    if missing:
        ok = False
        print(f"  ERROR: {missing} image files listed in JSON are missing on disk")
    if mismatched_size:
        ok = False
        print(f"  ERROR: {mismatched_size} images have a size mismatch vs. the JSON record")

    # 2. noun_phrase consistency between annotations and their category.
    phrase_mismatches = 0
    per_class_counts: Counter[str] = Counter()
    for ann in coco["annotations"]:
        cat = categories.get(ann["category_id"])
        expected_phrase = cat["noun_phrase"] if cat else None
        actual_phrase = ann.get("noun_phrase")
        if actual_phrase != expected_phrase:
            phrase_mismatches += 1
        per_class_counts[actual_phrase or "<missing noun_phrase>"] += 1
    if phrase_mismatches:
        ok = False
        print(f"  ERROR: {phrase_mismatches} annotations have a noun_phrase that "
              "doesn't match their category - fix the labeling/export step before training")

    print("  annotations per noun_phrase:")
    for phrase, count in per_class_counts.most_common():
        print(f"    {phrase!r}: {count}")

    # 3. Positive/negative image balance (class imbalance early-warning).
    positive_image_ids = {ann["image_id"] for ann in coco["annotations"]}
    n_positive = len(positive_image_ids)
    n_total = len(images_by_id)
    n_negative = n_total - n_positive
    pos_frac = n_positive / n_total if n_total else 0.0
    print(f"  images with >=1 mask: {n_positive}/{n_total} ({pos_frac:.1%}), "
          f"negative images: {n_negative}")
    if pos_frac < 0.05 or pos_frac > 0.95:
        print("  WARNING: heavily skewed positive/negative image ratio - "
              "consider adjusting --collapse in prepare_dataset.py")

    # 4. Spot-check that decoded RLE area matches the recorded area.
    area_mismatches = 0
    for ann in coco["annotations"][:200]:
        decoded = mask_utils.decode(ann["segmentation"])
        if int(decoded.sum()) != ann["area"]:
            area_mismatches += 1
    if area_mismatches:
        ok = False
        print(f"  ERROR: {area_mismatches}/200 sampled annotations have an RLE area "
              "mismatch - segmentation encoding is corrupt")

    return ok


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Output dir from prepare_dataset.py")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.dataset_root)

    all_ok = True
    for split in ("train", "val"):
        ann_path = root / f"{split}.json"
        images_dir = root / "images" / split
        if not ann_path.exists():
            print(f"skipping {split}: {ann_path} not found")
            continue
        all_ok &= verify_split(ann_path, images_dir)

    print("\nDataset verification " + ("PASSED" if all_ok else "FAILED - see ERRORs above"))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
