"""Command line for the synthetic generator.

    python -m synthetic_generator.cli generate --out D:\\data\\synthetic_75 --blocks 400
    python -m synthetic_generator.cli preview --view overview --out preview.png
    python -m synthetic_generator.cli inspect D:\\data\\synthetic_75
    python -m synthetic_generator.cli compare D:\\data\\synthetic_75 D:\\data\\training_10_7_26
    python -m synthetic_generator.cli presets
    python -m synthetic_generator.cli dump-config --preset "Complex structure" --out complex.json

``generate --events`` prints one ``@@{json}`` line per event; the GUI reads them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DTYPES, PRESETS, GeneratorConfig, default_config, preset

EVENT_PREFIX = "@@"


def _load_config(args: argparse.Namespace) -> GeneratorConfig:
    if getattr(args, "config", None):
        cfg = GeneratorConfig.load(args.config)
    elif getattr(args, "preset", None):
        cfg = preset(args.preset)
    else:
        cfg = default_config()
    if getattr(args, "out", None) and args.command == "generate":
        cfg.output.out_dir = str(args.out)
    for name, path in (("blocks", "output.num_blocks"), ("seed", "output.seed"), ("workers", "output.workers"),
                       ("start_id", "output.start_id"), ("dtype", "output.dtype")):
        value = getattr(args, name, None)
        if value is not None:
            cfg.set(path, value)
    if getattr(args, "overwrite", False):
        cfg.output.overwrite = True
    return cfg


def cmd_generate(args: argparse.Namespace) -> int:
    from .dataset import generate_dataset

    cfg = _load_config(args)
    stop_file = Path(args.stop_file) if args.stop_file else None

    def emit(ev: dict) -> None:
        if args.events:
            print(EVENT_PREFIX + json.dumps(ev), flush=True)
            return
        kind = ev.get("event")
        if kind == "start":
            print(f"Writing {ev['total']} blocks ({ev['total'] * ev['ids_per_block']} ids) to {ev['out_dir']} "
                  f"with {ev['workers']} workers; {ev['skipped']} complete blocks skipped.", flush=True)
        elif kind == "block":
            eta = f", ETA {ev['eta']:.0f}s" if ev.get("eta") is not None else ""
            print(f"  block {ev['done']}/{ev['total']} -> ids {ev['ids'][0]}-{ev['ids'][1]}{eta}", flush=True)
        elif kind in ("done", "stopped"):
            print(f"{kind.capitalize()}: {ev['written']} blocks written in {ev['seconds']}s; next free id {ev['next_id']}.",
                  flush=True)

    try:
        generate_dataset(cfg, cfg.output.out_dir, emit, lambda: bool(stop_file and stop_file.exists()))
    except (ValueError, OSError) as exc:
        if args.events:
            print(EVENT_PREFIX + json.dumps({"event": "error", "message": str(exc)}), flush=True)
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    from .block import generate_block
    from .preview import VIEWS, render_view

    cfg = _load_config(args)
    block = generate_block(cfg, args.block, keep_geology=True, csd=True)
    if args.view not in VIEWS:
        print(f"Unknown view {args.view!r}; choose from {list(VIEWS)}", file=sys.stderr)
        return 2
    image, _ = render_view(block, cfg, args.view, replica=args.replica, slice_index=args.slice,
                           scale=args.scale or cfg.grid.scales[0], tile=args.tile)
    image.save(args.out)
    print(f"Saved {args.out} ({image.width}x{image.height})")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .checks import inspect_dataset

    report = inspect_dataset(args.root, args.sample)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Dataset: {report['root']}")
        for key, info in report.get("folders", {}).items():
            print(f"  {info['folder']:<26} {info['files']:>7} files  ids {info['first']}..{info['last']}")
        print(f"  ids present in every folder: {report.get('common_ids', 0)}")
        if "finetuner_pairs" in report:
            print(f"  Quick Fine-Tuner pairs (Seis/Fault): {report['finetuner_pairs']}")
        if "replicas_share_labels" in report:
            print(f"  labels repeated across replicas: {report['replicas_share_labels']}")
        for note in report.get("notes", []):
            print(f"  note: {note}")
        for key, value in sorted(report.get("stats", {}).items()):
            print(f"  {key:<24} {value:10.4f}")
        for problem in report.get("problems", []):
            print(f"  PROBLEM: {problem}")
    return 1 if report.get("problems") else 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .checks import compare

    rows = compare(args.generated, args.reference, args.sample)
    print(f"{'statistic':<26}{'reference':>12}{'generated':>12}")
    for key, ref, gen in rows:
        print(f"{key:<26}{ref:12.4f}{gen:12.4f}")
    return 0


def cmd_presets(_args: argparse.Namespace) -> int:
    for name in PRESETS:
        print(name)
    return 0


def cmd_dump(args: argparse.Namespace) -> int:
    cfg = preset(args.preset) if args.preset else default_config()
    cfg.save(args.out)
    print(f"Saved {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m synthetic_generator.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Write a dataset.")
    g.add_argument("--config", help="Settings JSON (from the GUI or dump-config).")
    g.add_argument("--preset", choices=list(PRESETS))
    g.add_argument("--out", help="Output folder (overrides the config).")
    g.add_argument("--blocks", type=int, help="Number of blocks (each block = slices x replicas ids).")
    g.add_argument("--seed", type=int)
    g.add_argument("--workers", type=int, help="Worker processes (0 = auto).")
    g.add_argument("--start-id", type=int, help="First id (-1 = continue after existing ids).")
    g.add_argument("--dtype", choices=DTYPES)
    g.add_argument("--overwrite", action="store_true", help="Rewrite blocks that already exist.")
    g.add_argument("--events", action="store_true", help="Machine-readable progress (for the GUI).")
    g.add_argument("--stop-file", help="Stop after the running blocks once this file exists.")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("preview", help="Render a preview image of one block.")
    v.add_argument("--config")
    v.add_argument("--preset", choices=list(PRESETS))
    v.add_argument("--seed", type=int)
    v.add_argument("--block", type=int, default=0)
    v.add_argument("--view", default="overview")
    v.add_argument("--replica", type=int, default=0)
    v.add_argument("--slice", type=int, default=0)
    v.add_argument("--scale", type=int)
    v.add_argument("--tile", type=int, default=320)
    v.add_argument("--out", default="synthetic_preview.png")
    v.set_defaults(func=cmd_preview)

    i = sub.add_parser("inspect", help="Validate a dataset folder and print statistics.")
    i.add_argument("root")
    i.add_argument("--sample", type=int, default=120)
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=cmd_inspect)

    c = sub.add_parser("compare", help="Compare statistics of two dataset folders.")
    c.add_argument("generated")
    c.add_argument("reference")
    c.add_argument("--sample", type=int, default=120)
    c.set_defaults(func=cmd_compare)

    sub.add_parser("presets", help="List presets.").set_defaults(func=cmd_presets)
    d = sub.add_parser("dump-config", help="Write a settings JSON to edit by hand.")
    d.add_argument("--preset", choices=list(PRESETS))
    d.add_argument("--out", default="synthetic_config.json")
    d.set_defaults(func=cmd_dump)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
