"""Write blocks in the reference layout: one folder per array, ``<id>.dat`` per sample.

Every file is a raw little-endian float64 (or float32) ``H x W`` image, exactly
what ``finetuning.quick_data.read_section`` reads. With the default suffix the
folders are ``Seis75``, ``Fault75``, ``TauAtT75``, ``dipXscale{S}_75``,
``wtXscale{S}_75``, ``wtXStraddleScale{S}_75`` and ``wtXHaloScale{S}_75``.

Blocks are seeded by ``(seed, global block number)``, so a run can be resumed or
extended and always reproduces the same data. Metadata written next to the
folders: ``generator_config.json`` (settings of the latest run), ``runs.jsonl``
(one line per run), ``blocks.jsonl`` (geology/seismic parameters per block) and
``samples.csv`` (one row per id: block, replica, slice, bandpass, clip, ...).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .block import Block, Product, generate_block, products
from .config import GeneratorConfig

CONFIG_NAME = "generator_config.json"
RUNS_NAME = "runs.jsonl"
BLOCKS_NAME = "blocks.jsonl"
SAMPLES_NAME = "samples.csv"
SAMPLE_FIELDS = ("id", "block", "replica", "slice", "center", "center_hz", "clip", "clipped", "snr",
                 "fault_fraction", "faults_visible")
ID_LAYOUT = "id = start_id + block * (replicas * slices) + replica * slices + slice"

EventFn = Callable[[dict], None]


def item_bytes(cfg: GeneratorConfig) -> int:
    return 8 if cfg.output.dtype == "float64" else 4


def file_bytes(cfg: GeneratorConfig) -> int:
    return cfg.grid.height * cfg.grid.width * item_bytes(cfg)


def block_bytes(cfg: GeneratorConfig) -> int:
    return len(products(cfg)) * cfg.ids_per_block * file_bytes(cfg)


def dataset_bytes(cfg: GeneratorConfig) -> int:
    return block_bytes(cfg) * cfg.output.num_blocks


def folder_map(cfg: GeneratorConfig) -> dict[str, str]:
    return {f"{p.kind}{'' if p.scale is None else p.scale}": p.folder(cfg.output.suffix) for p in products(cfg)}


def existing_ids(out_dir: Path, cfg: GeneratorConfig) -> set[int]:
    ids: set[int] = set()
    for prod in products(cfg):
        folder = out_dir / prod.folder(cfg.output.suffix)
        if folder.is_dir():
            with os.scandir(folder) as it:
                for entry in it:
                    stem, ext = os.path.splitext(entry.name)
                    if ext.lower() == ".dat" and stem.isdigit():
                        ids.add(int(stem))
    return ids


def next_start_id(out_dir: Path, cfg: GeneratorConfig) -> int:
    ids = existing_ids(out_dir, cfg) if out_dir.is_dir() else set()
    if not ids:
        return 0
    per = cfg.ids_per_block
    return ((max(ids) + 1 + per - 1) // per) * per


def resolve_start_id(cfg: GeneratorConfig, out_dir: Path) -> int:
    return cfg.output.start_id if cfg.output.start_id >= 0 else next_start_id(out_dir, cfg)


def block_id_list(cfg: GeneratorConfig, start_id: int, offset: int) -> list[tuple[int, int, int]]:
    R, N = cfg.grid.n_replicas, cfg.grid.n_slices
    base = start_id + offset * R * N
    return [(base + r * N + s, r, s) for r in range(R) for s in range(N)]


def scan_sizes(out_dir: Path, cfg: GeneratorConfig) -> list[dict[int, int]]:
    """``{id: file size}`` for every product folder (one directory listing per folder)."""
    sizes: list[dict[int, int]] = []
    for prod in products(cfg):
        folder = out_dir / prod.folder(cfg.output.suffix)
        found: dict[int, int] = {}
        if folder.is_dir():
            with os.scandir(folder) as it:
                for entry in it:
                    stem, ext = os.path.splitext(entry.name)
                    if ext.lower() == ".dat" and stem.isdigit():
                        try:
                            found[int(stem)] = entry.stat().st_size
                        except OSError:
                            pass
        sizes.append(found)
    return sizes


def plan_blocks(out_dir: Path, cfg: GeneratorConfig, start_id: int) -> tuple[list[int], int, int]:
    """Blocks to write, blocks already complete (skipped) and bytes of files that do not exist yet."""
    size = file_bytes(cfg)
    sizes = scan_sizes(out_dir, cfg) if out_dir.is_dir() else [{} for _ in products(cfg)]
    pending: list[int] = []
    skipped = 0
    missing = 0
    for offset in range(cfg.output.num_blocks):
        ids = [sid for sid, _, _ in block_id_list(cfg, start_id, offset)]
        absent = sum(1 for found in sizes for sid in ids if sid not in found)
        complete = all(found.get(sid) == size for found in sizes for sid in ids)
        if complete and not cfg.output.overwrite:
            skipped += 1
            continue
        pending.append(offset)
        missing += absent
    return pending, skipped, missing * size


def write_block(out_dir: Path, cfg: GeneratorConfig, block: Block, start_id: int, offset: int) -> list[int]:
    dtype = "<f8" if cfg.output.dtype == "float64" else "<f4"
    ids = block_id_list(cfg, start_id, offset)
    for prod in products(cfg):
        folder = out_dir / prod.folder(cfg.output.suffix)
        folder.mkdir(parents=True, exist_ok=True)
        for sid, r, s in ids:
            np.ascontiguousarray(block.array(prod, r, s), dtype=dtype).tofile(folder / f"{sid}.dat")
    return [sid for sid, _, _ in ids]


def sample_rows(block: Block, cfg: GeneratorConfig, start_id: int, offset: int) -> list[dict]:
    seis = block.meta["seismic"]
    rows = []
    for sid, r, s in block_id_list(cfg, start_id, offset):
        rows.append({
            "id": sid, "block": block.index, "replica": r, "slice": s,
            "center": seis["centers"][r], "center_hz": seis["centers_hz"][r],
            "clip": seis["clips"][r], "clipped": int(seis["clipped"][r]), "snr": seis["snr"][r],
            "fault_fraction": block.meta["fault_fraction"][s],
            "faults_visible": block.meta["faults_visible"][s],
        })
    return rows


def _work(payload: dict) -> dict:
    cfg = GeneratorConfig.from_dict(payload["config"])
    t0 = time.perf_counter()
    block = generate_block(cfg, payload["block_index"])
    ids = write_block(Path(payload["out_dir"]), cfg, block, payload["start_id"], payload["offset"])
    return {
        "offset": payload["offset"],
        "block_index": payload["block_index"],
        "ids": [ids[0], ids[-1]],
        "seconds": round(time.perf_counter() - t0, 3),
        "meta": block.meta,
        "rows": sample_rows(block, cfg, payload["start_id"], payload["offset"]),
    }


def auto_workers(cfg: GeneratorConfig) -> int:
    if cfg.output.workers > 0:
        return cfg.output.workers
    return max(1, min(8, (os.cpu_count() or 2) - 1))


# One full default block (15 arrays, 144 MB) wrote at about this rate with
# ``ndarray.tofile``, the same call the generator uses. The GUI replaces it
# with a measurement of the output drive when it can.
DEFAULT_WRITE_BPS = 200_000_000

# Spawning the process pool (each worker imports this package) measured about
# 0.9s plus 0.12s per worker on Windows. One worker stays in-process, so it
# pays no startup. Checked against timed runs of the default preset:
# 8 blocks / 1 worker 11.6s, 8 / 4 workers 4.4s, 8 / 8 workers 3.3s.
_POOL_START_S = 0.9
_POOL_START_PER_WORKER_S = 0.12


def block_cpu_weight(cfg: GeneratorConfig, *, csd: bool | None = None) -> float:
    """Relative CPU cost of building one block. Only ratios between configs are used.

    Timed ``generate_block`` calls track the section area. Each slice pays for
    the geology, reflectivity and labels; each replica pays for filtering the
    wavelet, and that part grows with the supersample. Tau-mapping filters one
    impedance log instead of a supersampled section. Halo blurs get wider as the
    scale S grows, so the label cost follows both the number of scales and their
    sum. CSD adds a smaller pass over the same grid.
    """
    grid = cfg.grid
    area = max(1, int(grid.height)) * max(1, int(grid.width))
    scales = [max(1, int(s)) for s in grid.scales] or [1]
    scale_units = 0.62 * (len(scales) / 3) + 0.38 * (sum(scales) / 108)
    per_slice = 0.73 + 0.27 * scale_units
    if cfg.seismic.render_mode == "convolution":
        per_replica = 0.131 * max(1, int(cfg.seismic.supersample)) / 4
    else:
        per_replica = 0.036
    slices = max(1, int(grid.n_slices))
    replicas = max(1, int(grid.n_replicas))
    weight = area * slices * (per_slice + replicas * per_replica)
    include_csd = cfg.csd.enabled if csd is None else csd
    if include_csd:
        horizons = max(1.0, 0.5 * float(sum(cfg.csd.count)))
        weight += area * slices * (0.22 + 0.03 * horizons)
    return float(weight)


def estimate_run_seconds(cfg: GeneratorConfig, block_compute_s: float, *,
                         write_bps: float = DEFAULT_WRITE_BPS, n_blocks: int | None = None) -> float:
    """Wall-clock seconds to build and write ``n_blocks`` (the configured count by default).

    ``block_compute_s`` is one ``generate_block`` on this machine, usually the
    live preview. The worker then writes that block, and timed runs show those
    two steps add. Extra workers split the blocks evenly (compute and writing
    both sped up on the machine this was checked on). A process pool pays a
    small startup; a single worker does not, because it stays in-process.
    """
    n = cfg.output.num_blocks if n_blocks is None else int(n_blocks)
    if n <= 0:
        return 0.0
    workers = min(auto_workers(cfg), n)
    per_block = max(0.0, float(block_compute_s)) + block_bytes(cfg) / max(float(write_bps), 1.0)
    if workers == 1:
        return n * per_block
    return _POOL_START_S + _POOL_START_PER_WORKER_S * workers + n * per_block / workers


def measure_write_bps(directory: str | Path, n_files: int = 40) -> float:
    """Bytes per second writing section files the way a run does. The files are removed."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=".wprobe_", dir=root))
    blob = np.random.default_rng(0).random(224 * 224).astype("<f8")
    try:
        t0 = time.perf_counter()
        for i in range(n_files):
            blob.tofile(folder / f"{i}.dat")
        elapsed = time.perf_counter() - t0
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return n_files * int(blob.nbytes) / max(elapsed, 1e-6)


def check_disk(out_dir: Path, needed: int) -> tuple[bool, int]:
    probe = out_dir
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    return free >= needed * 1.02, free


def generate_dataset(cfg: GeneratorConfig, out_dir: str | Path | None = None, on_event: EventFn | None = None,
                     should_stop: Callable[[], bool] | None = None) -> dict:
    """Generate ``cfg.output.num_blocks`` blocks into ``out_dir``; returns a run summary."""
    emit = on_event or (lambda ev: None)
    stop = should_stop or (lambda: False)
    errors = cfg.validate()
    if errors:
        raise ValueError("Invalid settings:\n- " + "\n- ".join(errors))
    target = cfg.output.out_dir if out_dir is None else out_dir
    if isinstance(target, str) and not target.strip():
        raise ValueError("No output folder given.")
    out = Path(target)
    out.mkdir(parents=True, exist_ok=True)
    start_id = resolve_start_id(cfg, out)
    if start_id % cfg.ids_per_block:
        raise ValueError(f"Start id {start_id} must be a multiple of {cfg.ids_per_block} (ids per block).")
    first_block = start_id // cfg.ids_per_block
    prods = products(cfg)
    for prod in prods:
        (out / prod.folder(cfg.output.suffix)).mkdir(parents=True, exist_ok=True)

    pending, skipped, needed = plan_blocks(out, cfg, start_id)
    ok, free = check_disk(out, needed)
    if not ok:
        raise OSError(f"Not enough disk space: need {needed / 1e9:.2f} GB, {free / 1e9:.2f} GB free on the target drive.")

    started = datetime.now()
    record = cfg.to_dict()
    record["run"] = {
        "generator_version": __version__, "created": started.isoformat(timespec="seconds"),
        "out_dir": str(out), "start_id": start_id, "first_block": first_block,
        "ids_per_block": cfg.ids_per_block, "id_layout": ID_LAYOUT, "folders": folder_map(cfg),
    }
    (out / CONFIG_NAME).write_text(json.dumps(record, indent=2), encoding="utf-8")

    workers = min(auto_workers(cfg), max(1, len(pending)))
    emit({"event": "start", "total": len(pending), "skipped": skipped, "start_id": start_id,
          "ids_per_block": cfg.ids_per_block, "workers": workers, "out_dir": str(out),
          "bytes": needed, "folders": [p.folder(cfg.output.suffix) for p in prods]})

    samples_path = out / SAMPLES_NAME
    new_samples = not samples_path.exists()
    cfg_dict = cfg.to_dict()
    t0 = time.perf_counter()
    done = 0
    stopped = False
    with open(samples_path, "a", newline="", encoding="utf-8") as sf, \
            open(out / BLOCKS_NAME, "a", encoding="utf-8") as bf:
        writer = csv.DictWriter(sf, fieldnames=SAMPLE_FIELDS)
        if new_samples:
            writer.writeheader()

        def handle(result: dict) -> None:
            nonlocal done
            done += 1
            writer.writerows(result["rows"])
            sf.flush()
            bf.write(json.dumps({"ids": result["ids"], **result["meta"]}) + "\n")
            bf.flush()
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            emit({"event": "block", "done": done, "total": len(pending), "block": result["block_index"],
                  "ids": result["ids"], "seconds": result["seconds"], "elapsed": round(elapsed, 2),
                  "eta": round((len(pending) - done) / rate, 1) if rate > 0 else None,
                  "faults": len(result["meta"]["geology"]["faults"]),
                  "centers_hz": result["meta"]["seismic"]["centers_hz"]})

        payloads = ({"config": cfg_dict, "out_dir": str(out), "start_id": start_id, "offset": o,
                     "block_index": first_block + o} for o in pending)
        if workers == 1:
            for payload in payloads:
                if stop():
                    stopped = True
                    break
                handle(_work(payload))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                inflight = set()
                exhausted = False
                while True:
                    while not exhausted and not stopped and len(inflight) < 2 * workers:
                        payload = next(payloads, None)
                        if payload is None:
                            exhausted = True
                            break
                        inflight.add(pool.submit(_work, payload))
                    if not inflight:
                        break
                    finished, inflight = wait(inflight, timeout=0.5, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        handle(fut.result())
                    if not stopped and stop():
                        stopped = True
                        emit({"event": "stopping", "inflight": len(inflight)})

    seconds = round(time.perf_counter() - t0, 2)
    summary = {"event": "stopped" if stopped else "done", "written": done, "skipped": skipped,
               "start_id": start_id, "next_id": start_id + cfg.output.num_blocks * cfg.ids_per_block,
               "seconds": seconds, "out_dir": str(out)}
    with open(out / RUNS_NAME, "a", encoding="utf-8") as rf:
        rf.write(json.dumps({**record["run"], "blocks": cfg.output.num_blocks, "seed": cfg.output.seed,
                             "written": done, "skipped": skipped, "stopped": stopped, "seconds": seconds}) + "\n")
    emit(summary)
    return summary


def read_block_from_disk(out_dir: Path, cfg: GeneratorConfig, start_id: int, offset: int) -> dict[Product, np.ndarray]:
    """Load one block back as ``{product: (R, N, H, W)}`` (used by tests and the inspector)."""
    dtype = "<f8" if cfg.output.dtype == "float64" else "<f4"
    R, N, H, W = cfg.grid.n_replicas, cfg.grid.n_slices, cfg.grid.height, cfg.grid.width
    out: dict[Product, np.ndarray] = {}
    for prod in products(cfg):
        arr = np.empty((R, N, H, W))
        folder = out_dir / prod.folder(cfg.output.suffix)
        for sid, r, s in block_id_list(cfg, start_id, offset):
            arr[r, s] = np.fromfile(folder / f"{sid}.dat", dtype=dtype).reshape(H, W)
        out[prod] = arr
    return out
