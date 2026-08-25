"""Reproducible, non-Streamlit benchmark for interactive SAM 3 inference.

The benchmark intentionally uses the application's public loader,
preprocessing, point-segmentation, and propagation APIs. Model loading is
reported separately and is not included in inference timings.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seismic_app import config  # noqa: E402
from seismic_app.inference import Sam3PointSegmenter, Sam3VolumePropagator  # noqa: E402
from seismic_app.preprocessing import (  # noqa: E402
    inline_to_rgb_25d,
    normalize_to_uint8,
    to_rgb,
)
from seismic_app.sgy_loader import load_any  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SAM 3 point segmentation and 3D propagation."
    )
    parser.add_argument("sgy", type=Path, help="Path to a 2D or 3D SEG-Y file.")
    parser.add_argument(
        "--checkpoint", default=config.DEFAULT_CHECKPOINT, help="SAM 3 checkpoint."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, cuda, or a CUDA device such as cuda:0.",
    )
    parser.add_argument(
        "--axis",
        choices=("inline", "crossline", "time"),
        default="inline",
        help="Slice axis used for point inference and 3D propagation.",
    )
    parser.add_argument(
        "--anchor",
        type=int,
        default=None,
        help="Anchor slice index (default: middle slice).",
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=5,
        help="Number of cached point and repeated propagation runs (default: 5).",
    )
    parser.add_argument(
        "--mode",
        choices=("point", "propagation", "both"),
        default="both",
        help="Workload to run (default: both).",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Use float32 for propagation instead of CUDA bfloat16.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Benchmark torch.compile for the video tracker (long first warm-up).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Benchmark a centered subset of propagation frames.",
    )
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def timed(device: str, operation: Callable[[], Any]) -> tuple[Any, float]:
    synchronize(device)
    started = time.perf_counter()
    value = operation()
    synchronize(device)
    return value, time.perf_counter() - started


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def timing_summary(seconds: list[float]) -> dict[str, Any]:
    return {
        "runs": len(seconds),
        "seconds": seconds,
        "mean_seconds": statistics.fmean(seconds) if seconds else None,
        "p50_seconds": percentile(seconds, 50),
        "p95_seconds": percentile(seconds, 95),
    }


def resolve_device(requested: str) -> tuple[str, str | None]:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda", None
        return "cpu", "CUDA is unavailable; auto selected CPU."
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu", f"{requested} was requested, but CUDA is unavailable; using CPU."
    return requested, None


def frame_count(data: np.ndarray, axis: str) -> int:
    return {
        "inline": data.shape[0],
        "crossline": data.shape[1],
        "time": data.shape[2],
    }[axis]


def make_frame(data_u8: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "inline":
        return inline_to_rgb_25d(data_u8, index)
    if axis == "crossline":
        return to_rgb(data_u8[:, index, :].T)
    return to_rgb(data_u8[:, :, index])


def cuda_memory(device: str) -> dict[str, Any]:
    if not device.startswith("cuda"):
        return {
            "available": False,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    cuda_device = torch.device(device)
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(cuda_device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(cuda_device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(cuda_device),
    }


def reset_cuda_peak(device: str) -> None:
    if device.startswith("cuda"):
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def main() -> int:
    args = parse_args()
    if args.warm_runs < 1:
        raise SystemExit("--warm-runs must be at least 1")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device, device_note = resolve_device(args.device)
    report: dict[str, Any] = {
        "configuration": {
            "sgy": str(args.sgy.resolve()),
            "checkpoint": args.checkpoint,
            "requested_device": args.device,
            "resolved_device": device,
            "axis": args.axis,
            "requested_anchor": args.anchor,
            "warm_runs": args.warm_runs,
            "mode": args.mode,
            "propagation_dtype": "float32" if args.float32 else "bfloat16",
            "compile": args.compile,
            "max_frames": args.max_frames,
            "seed": 0,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "device_note": device_note,
    }

    load_started = time.perf_counter()
    data, geometry = load_any(args.sgy)
    data_u8 = normalize_to_uint8(data)
    report["data"] = {
        "kind": geometry.kind,
        "shape": list(data.shape),
        "load_and_normalize_seconds": time.perf_counter() - load_started,
    }

    if geometry.kind == "2d":
        anchor = 0
        rgb = to_rgb(data_u8)
    else:
        count = frame_count(data, args.axis)
        anchor = count // 2 if args.anchor is None else args.anchor
        if not 0 <= anchor < count:
            raise SystemExit(f"--anchor must be in [0, {count - 1}] for {args.axis}")
        rgb = make_frame(data_u8, args.axis, anchor)
    point = (rgb.shape[1] // 2, rgb.shape[0] // 2)
    report["configuration"]["anchor"] = anchor
    report["configuration"]["point_xy"] = list(point)

    if args.mode in ("point", "both"):
        load_started = time.perf_counter()
        point_model = Sam3PointSegmenter(args.checkpoint, device=device)
        synchronize(device)
        model_load_seconds = time.perf_counter() - load_started
        reset_cuda_peak(device)

        _, cold_seconds = timed(
            device, lambda: point_model.segment(rgb, [point], [1])
        )
        warm_seconds = [
            timed(device, lambda: point_model.segment(rgb, [point], [1]))[1]
            for _ in range(args.warm_runs)
        ]
        report["point"] = {
            "model_load_seconds": model_load_seconds,
            "cold_seconds": cold_seconds,
            "warm": timing_summary(warm_seconds),
            "last_stage_seconds": dict(point_model.last_timings),
            "peak_cuda_vram": cuda_memory(device),
        }
        del point_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.mode in ("propagation", "both"):
        if geometry.kind != "3d":
            report["propagation"] = {
                "status": "skipped",
                "reason": "Propagation requires a 3D SEG-Y volume.",
                "prep_seconds": None,
                "session_seconds": None,
                "inference_seconds": None,
                "total_seconds": None,
                "fps": None,
            }
        else:
            prep_started = time.perf_counter()
            full_count = frame_count(data, args.axis)
            start = 0
            stop = full_count
            if args.max_frames is not None:
                if args.max_frames < 2:
                    raise SystemExit("--max-frames must be at least 2")
                subset = min(args.max_frames, full_count)
                start = max(0, min(anchor - subset // 2, full_count - subset))
                stop = start + subset
            frames = [
                make_frame(data_u8, args.axis, i) for i in range(start, stop)
            ]
            count = len(frames)
            propagation_anchor = anchor - start
            prep_seconds = time.perf_counter() - prep_started

            load_started = time.perf_counter()
            propagator = Sam3VolumePropagator(
                args.checkpoint,
                device=device,
                use_bfloat16=not args.float32,
                compile_model=args.compile,
            )
            synchronize(device)
            model_load_seconds = time.perf_counter() - load_started
            reset_cuda_peak(device)

            def propagate() -> np.ndarray:
                return propagator.propagate(
                    frames, propagation_anchor, [[point]], [[1]]
                )

            _, cold_total = timed(device, propagate)
            cold_stages = dict(propagator.last_timings)
            warm_totals = []
            warm_stages = []
            for _ in range(args.warm_runs):
                warm_totals.append(timed(device, propagate)[1])
                warm_stages.append(dict(propagator.last_timings))
            report["propagation"] = {
                "status": "completed",
                "frames": count,
                "source_frame_range": [start, stop],
                "model_load_seconds": model_load_seconds,
                "prep_seconds": prep_seconds,
                "cold_stage_seconds": cold_stages,
                "cold_total_seconds": cold_total,
                "cold_fps": count / cold_total,
                "warm_total": timing_summary(warm_totals),
                "warm_fps": timing_summary([count / value for value in warm_totals]),
                "warm_stage_seconds": warm_stages,
                "peak_cuda_vram": cuda_memory(device),
            }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
