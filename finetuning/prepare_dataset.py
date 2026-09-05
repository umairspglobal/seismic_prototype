"""Step 1 of the SAM 3 fine-tuning workflow: turn the raw 3D training cubes
into a 2D COCO-format dataset that SAM 3's training repo (and the
verification/validation scripts in this folder) can consume directly.

Why this script looks the way it does
--------------------------------------
SAM 3 (and this app's inference pipeline in seismic_app/) works on 2D
images, but the training data on disk is a set of raw float64 binary cubes:

- ``Seis80/<id>.dat``   -> one (256, 256, 256) seismic amplitude cube.
- ``ImpMask80/<id>.dat``-> one (256, 256, 256, 20) mask cube: 20 binary
  channels **last** (verified by comparing candidate axis orderings against
  the seismic structure at a matching slice - a bins-first reshape produces
  spatially-incoherent vertical-stripe noise uncorrelated with the seismic
  section's folds, while bins-last produces a smooth mask that follows the
  same folds). Each channel is an ordered low->high acoustic-impedance band:
  values are strictly {0, 1}, channels are NOT one-hot, and the per-channel
  "on" fraction rises then falls smoothly across the 20 channels, consistent
  with overlapping quantile-style impedance bins rather than 20 unrelated
  geological classes.

This app's fixed SAM 3 vocabulary (seismic_app/config.py) has no per-bin
concept, so the 20 channels are collapsed into a single binary mask for the
"seismic facies" noun_phrase (as decided with the user: this dataset only
covers that one class - fault/channel/salt body/horizon are left zero-shot
for now). Default collapse = union of all 20 bins ("is this voxel part of
ANY impedance-classified facies region"), which measured ~75% positive
coverage on the sample cube inspected - if that's not the intended
semantics, rerun with --collapse bin-range and narrower --min-bin/--max-bin.

Cubes are sliced along the inline axis (axis 0), matching the axis
convention seismic_app/preprocessing.inline_to_rgb_25d uses for real 3D
volumes, and reuse that exact function (plus normalize_to_uint8) so the
fine-tuning images are preprocessed identically to what the app feeds SAM 3
at inference time. The split is 85/15 by CUBE id (not by slice), since
slices from the same cube are highly correlated and splitting by slice
would leak information into validation.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from seismic_app.preprocessing import inline_to_rgb_25d, normalize_to_uint8

try:
    from pycocotools import mask as mask_utils
except ImportError as exc:  # pragma: no cover - environment setup issue
    raise SystemExit(
        "pycocotools is required: pip install -r finetuning/requirements-finetune.txt"
    ) from exc

NOUN_PHRASE = "seismic facies"
CATEGORY_ID = 1


@dataclass
class CubePaths:
    cube_id: int
    seis_path: Path
    mask_path: Path


def _discover_cubes(seis_dir: Path, mask_dir: Path) -> list[CubePaths]:
    """Match Seis80/<id>.dat <-> ImpMask80/<id>.dat by numeric id."""
    cubes: list[CubePaths] = []
    for seis_path in sorted(seis_dir.glob("*.dat"), key=lambda p: int(p.stem)):
        mask_path = mask_dir / seis_path.name
        if not mask_path.exists():
            print(f"  skipping cube {seis_path.stem}: no matching mask file")
            continue
        cubes.append(CubePaths(int(seis_path.stem), seis_path, mask_path))
    if not cubes:
        raise FileNotFoundError(
            f"No matching *.dat pairs found between {seis_dir} and {mask_dir}"
        )
    return cubes


def load_seismic_cube(path: Path, cube_size: int) -> np.ndarray:
    """Read a raw float64 seismic cube, shape (cube_size, cube_size, cube_size)."""
    flat = np.fromfile(path, dtype="<f8")
    expected = cube_size**3
    if flat.size != expected:
        raise ValueError(
            f"{path}: expected {expected} float64 elements ({cube_size}^3), got {flat.size}"
        )
    return flat.reshape(cube_size, cube_size, cube_size)


def load_mask_cube(path: Path, cube_size: int, num_bins: int) -> np.ndarray:
    """Read a raw float64 mask cube, shape (cube_size, cube_size, cube_size, num_bins).

    The bins axis is LAST, not first - verified by comparing candidate
    reshapes against the seismic structure at a matching slice (a
    bins-first reshape produces spatially-incoherent vertical-stripe
    noise uncorrelated with the seismic section's folds; this layout
    produces a smooth mask that follows the same folds).
    """
    flat = np.fromfile(path, dtype="<f8")
    expected = num_bins * cube_size**3
    if flat.size != expected:
        raise ValueError(
            f"{path}: expected {expected} float64 elements "
            f"({num_bins} x {cube_size}^3), got {flat.size}"
        )
    return flat.reshape(cube_size, cube_size, cube_size, num_bins)


def collapse_mask(
    mask_cube: np.ndarray,
    mode: str = "union",
    min_bin: int = 0,
    max_bin: int | None = None,
) -> np.ndarray:
    """Collapse the (D, H, W, num_bins) impedance-bin cube to one binary (D, H, W) mask."""
    if mode == "union":
        return mask_cube.any(axis=-1)
    if mode == "intersection":
        return mask_cube.all(axis=-1)
    if mode == "bin-range":
        hi = mask_cube.shape[-1] if max_bin is None else max_bin + 1
        return mask_cube[..., min_bin:hi].any(axis=-1)
    raise ValueError(f"Unknown collapse mode {mode!r}")


def _rle_annotation(binary_mask: np.ndarray, ann_id: int, image_id: int) -> dict:
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    rle["counts"] = rle["counts"].decode("ascii")
    ys, xs = np.where(binary_mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": CATEGORY_ID,
        "noun_phrase": NOUN_PHRASE,
        "segmentation": rle,
        "area": int(binary_mask.sum()),
        "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
        "iscrowd": 0,
    }


def _write_split(
    cubes: list[CubePaths],
    out_dir: Path,
    split: str,
    cube_size: int,
    num_bins: int,
    collapse_mode: str,
    min_bin: int,
    max_bin: int | None,
    slice_stride: int,
) -> None:
    img_dir = out_dir / "images" / split
    mask_dir_out = out_dir / "masks" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir_out.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 1
    ann_id = 1
    n_positive = 0

    for cube in cubes:
        seis = load_seismic_cube(cube.seis_path, cube_size)
        mask = load_mask_cube(cube.mask_path, cube_size, num_bins)
        seis_uint8 = normalize_to_uint8(seis)
        mask_bool = collapse_mask(mask, collapse_mode, min_bin, max_bin)
        del mask

        for iline in range(0, cube_size, slice_stride):
            rgb = inline_to_rgb_25d(seis_uint8, iline)  # (n_samples, n_xlines, 3)
            mask_slice = mask_bool[iline].T  # (xline, sample) -> (sample, xline)

            name = f"cube{cube.cube_id:03d}_il{iline:04d}.png"
            Image.fromarray(rgb).save(img_dir / name)
            Image.fromarray((mask_slice * 255).astype(np.uint8)).save(mask_dir_out / name)

            h, w = mask_slice.shape
            images.append(
                {"id": image_id, "file_name": name, "height": h, "width": w}
            )
            if mask_slice.any():
                annotations.append(_rle_annotation(mask_slice, ann_id, image_id))
                ann_id += 1
                n_positive += 1
            image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": CATEGORY_ID, "name": NOUN_PHRASE, "noun_phrase": NOUN_PHRASE}
        ],
    }
    ann_path = out_dir / f"{split}.json"
    ann_path.write_text(json.dumps(coco))
    print(
        f"{split}: {len(cubes)} cubes -> {len(images)} images, "
        f"{n_positive} with a positive mask ({len(images) - n_positive} negatives) "
        f"-> {ann_path}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seis-dir", required=True, help="Path to the Seis80 folder.")
    parser.add_argument("--mask-dir", required=True, help="Path to the ImpMask80 folder.")
    parser.add_argument("--out-dir", required=True, help="Output dataset root.")
    parser.add_argument("--cube-size", type=int, default=256)
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument(
        "--collapse",
        choices=["union", "intersection", "bin-range"],
        default="union",
        help="How to collapse the 20 impedance bins into one binary mask (default: union).",
    )
    parser.add_argument("--min-bin", type=int, default=0, help="Used with --collapse bin-range.")
    parser.add_argument("--max-bin", type=int, default=None, help="Used with --collapse bin-range.")
    parser.add_argument(
        "--slice-stride",
        type=int,
        default=2,
        help="Take every Nth inline slice per cube (default: 2, i.e. 128 slices/cube). "
        "Use 1 for the full 256 slices/cube.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-cubes",
        type=int,
        default=None,
        help="After shuffling, keep at most this many cubes (then split 85/15). "
        "Use a subset so Phase 1 can finish in a same-day budget.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seis_dir = Path(args.seis_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)

    cubes = _discover_cubes(seis_dir, mask_dir)
    print(f"Found {len(cubes)} matched seismic/mask cube pairs.")

    rng = random.Random(args.seed)
    shuffled = cubes[:]
    rng.shuffle(shuffled)
    if args.max_cubes is not None:
        if args.max_cubes < 2:
            raise SystemExit("--max-cubes must be at least 2 so both train and val have a cube.")
        shuffled = shuffled[: args.max_cubes]
        print(f"Subsampled to {len(shuffled)} cubes (--max-cubes {args.max_cubes}).")
    n_val = max(1, round(len(shuffled) * args.val_fraction))
    n_val = min(n_val, len(shuffled) - 1)
    val_cubes, train_cubes = shuffled[:n_val], shuffled[n_val:]
    print(f"Split: {len(train_cubes)} train cubes, {len(val_cubes)} val cubes.")

    for split, split_cubes in [("train", train_cubes), ("val", val_cubes)]:
        _write_split(
            split_cubes,
            out_dir,
            split,
            args.cube_size,
            args.num_bins,
            args.collapse,
            args.min_bin,
            args.max_bin,
            args.slice_stride,
        )


if __name__ == "__main__":
    main()
