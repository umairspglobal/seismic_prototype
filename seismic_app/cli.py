"""Command-line entry point.

Usage:
    python -m seismic_app.cli data/1.sgy --out outputs/
    python -m seismic_app.cli data/*.sgy --out outputs/ --device cpu
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from . import config
from .inference import Sam3SeismicSegmenter
from .pipeline import process_and_export


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatically segment faults, channels, facies, salt "
        "bodies, and horizons in a 2D .sgy seismic section using SAM 3 - "
        "no interactive prompts required."
    )
    parser.add_argument(
        "sgy_paths",
        nargs="+",
        help="Path(s) to .sgy file(s). Glob patterns are expanded.",
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output directory for overlay PNGs and mask .npz files (default: outputs/).",
    )
    parser.add_argument(
        "--checkpoint",
        default=config.DEFAULT_CHECKPOINT,
        help="SAM 3 checkpoint to load (default: facebook/sam3). Requires "
        "accepting the license at https://huggingface.co/facebook/sam3 and "
        "running `hf auth login` (or setting HF_TOKEN) beforehand.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. 'cuda' or 'cpu' (default: auto-detect).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.MASK_THRESHOLD,
        help=f"Probability threshold for binarizing masks (default: {config.MASK_THRESHOLD}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    paths: list[str] = []
    for pattern in args.sgy_paths:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])

    if not paths:
        print("No .sgy files matched.", file=sys.stderr)
        return 1

    print(f"Loading SAM 3 checkpoint '{args.checkpoint}'...")
    segmenter = Sam3SeismicSegmenter(checkpoint=args.checkpoint, device=args.device)
    print(f"Using device: {segmenter.device}")

    out_dir = Path(args.out)
    for path in paths:
        print(f"Processing {path} ...")
        try:
            process_and_export(
                path,
                out_dir,
                segmenter=segmenter,
                threshold=args.threshold,
            )
        except Exception as exc:  # noqa: BLE001 - surface per-file failures, keep going
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        print(f"  wrote {out_dir / (Path(path).stem + '_overlay.png')}")
        print(f"  wrote {out_dir / (Path(path).stem + '_masks.npz')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
