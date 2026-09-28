"""One-time conversion of large seismic files into memory-mapped uint8 caches.

A multi-GB SEG-Y cannot be loaded into RAM, normalized and sliced the
way small files are. Instead it is streamed once through
``build_cache``:

1. **scanning** - exact trace -> grid cell map from the header pair
   detected by ``segy_geometry`` (irregular grids and either sort order);
2. **sampling** - clip percentiles from ~20k evenly spaced traces, the
   same ``CLIP_LOW/HIGH_PERCENTILE`` as ``normalize_to_uint8``;
3. **converting** - blocks of ~256 MB are decoded, clipped/scaled to
   uint8 in place and written to three axis-ordered memmaps
   (``inline.u8`` (n_il, n_xl, n_s), ``crossline.u8`` (n_xl, n_il, n_s),
   ``time.u8`` (n_s, n_il, n_xl)), so a slice along any axis is one
   contiguous read. 2D lines get a single ``section.u8`` (n_s, n_traces).

Files are written as ``*.partial`` and renamed when complete; the
manifest is written last, so an interrupted build is simply redone.
``CacheManager`` runs builds one at a time in a child process, so the
decode work never competes with the API process for the GIL or the GPU.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable

import numpy as np

from . import config, sysinfo
from .geometry import SectionGeometry
from .logutil import get_logger
from .segy_geometry import GridLayout, cells_for_traces, scan_layout_exact
from .sources import SegySource, SeismicSource, open_source

log = get_logger("volume_cache")

CACHE_VERSION = 1
ORDERS_3D = ("inline", "crossline", "time")
Emit = Callable[[dict], None]


# ---- keys, manifests, geometry persistence --------------------------------


def cache_key(path: str | Path) -> str:
    """Stable key for one version of a file: name, size, mtime and edge bytes."""
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha1(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    with open(path, "rb") as fh:
        digest.update(fh.read(65536))
        if stat.st_size > 65536:
            fh.seek(max(0, stat.st_size - 65536))
            digest.update(fh.read(65536))
    safe_stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in path.stem)
    return f"{safe_stem}-{stat.st_size}-{int(stat.st_mtime)}-{digest.hexdigest()[:12]}"


def cache_dir_for(path: str | Path, root: Path | None = None) -> Path:
    return Path(root or config.CACHE_DIR) / cache_key(path)


def order_shape(order: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 2:
        return shape
    n_il, n_xl, n_s = shape
    return {
        "inline": (n_il, n_xl, n_s),
        "crossline": (n_xl, n_il, n_s),
        "time": (n_s, n_il, n_xl),
    }[order]


def read_manifest(directory: Path) -> dict | None:
    """The manifest of a complete, current-version cache; None otherwise."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return None
    if manifest.get("version") != CACHE_VERSION or not manifest.get("complete"):
        return None
    for order in manifest.get("orders", []):
        expected = math.prod(order_shape(order, tuple(manifest["shape"])))
        target = directory / f"{order}.u8"
        if not target.is_file() or target.stat().st_size != expected:
            return None
    return manifest


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def save_geometry(path: Path, geometry: SectionGeometry) -> None:
    arrays: dict[str, np.ndarray] = {}
    for item in fields(SectionGeometry):
        value = getattr(geometry, item.name)
        if value is None:
            continue
        arrays[item.name] = np.asarray(value)
    np.savez(path, **arrays)


def load_geometry(path: Path) -> SectionGeometry:
    with np.load(path, allow_pickle=False) as data:
        kwargs: dict = {}
        for item in fields(SectionGeometry):
            if item.name not in data:
                continue
            value = data[item.name]
            if value.ndim == 0:
                scalar = value.item()
                kwargs[item.name] = scalar
            else:
                kwargs[item.name] = np.array(value)
    kwargs["kind"] = str(kwargs["kind"])
    for key in ("n_traces", "n_samples"):
        kwargs[key] = int(kwargs[key])
    return SectionGeometry(**kwargs)


def open_memmaps(directory: Path, manifest: dict) -> dict[str, np.memmap]:
    shape = tuple(manifest["shape"])
    return {
        order: np.memmap(
            directory / f"{order}.u8",
            dtype=np.uint8,
            mode="r",
            shape=order_shape(order, shape),
        )
        for order in manifest["orders"]
    }


def remove_stale_caches(root: Path, source_name: str, keep: Path) -> None:
    """Delete older caches of the same source file to reclaim disk space."""
    if not root.is_dir():
        return
    for child in root.iterdir():
        if not child.is_dir() or child.resolve() == keep.resolve():
            continue
        manifest_path = child / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        except (OSError, ValueError):
            manifest = {}
        if manifest.get("source") == source_name:
            log.info("Removing stale cache %s for %s", child.name, source_name)
            shutil.rmtree(child, ignore_errors=True)


# ---- normalization ----------------------------------------------------------


def clip_bounds(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(
        samples, [config.CLIP_LOW_PERCENTILE, config.CLIP_HIGH_PERCENTILE]
    )
    return float(lo), float(hi)


def normalize_block(block: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """In-place float32 clip/scale -> uint8, the same arithmetic as normalize_to_uint8."""
    if hi <= lo:
        return np.zeros(block.shape, dtype=np.uint8)
    lo32 = np.float32(lo)
    np.clip(block, lo32, np.float32(hi), out=block)
    block -= lo32
    block /= np.float32(hi - lo)
    block *= np.float32(255)
    return block.astype(np.uint8)


def fill_value(lo: float, hi: float) -> int:
    """uint8 level of a zero amplitude, used for missing traces."""
    return int(normalize_block(np.zeros(1, dtype=np.float32), lo, hi)[0])


# ---- the build --------------------------------------------------------------


def _close_memmap(mm: np.memmap) -> None:
    mm.flush()
    handle = getattr(mm, "_mmap", None)
    if handle is not None:
        try:
            handle.close()
        except (BufferError, ValueError):
            pass


def _choose_orders(kind: str, sort: str, u8_bytes: int, directory: Path) -> list[str]:
    if kind != "3d":
        needed = int(u8_bytes * 1.1)
        if sysinfo.free_disk(directory) < needed:
            raise OSError(
                f"Not enough free disk for the cache: need {sysinfo.gb(needed)}, "
                f"have {sysinfo.gb(sysinfo.free_disk(directory))} at {directory}"
            )
        return ["section"]
    free = sysinfo.free_disk(directory)
    full = int(3 * u8_bytes * 1.1)
    log.info(
        "Disk check at %s: %s free, full cache needs %s",
        directory,
        sysinfo.gb(free),
        sysinfo.gb(full),
    )
    if free >= full:
        return list(ORDERS_3D)
    primary = "inline" if sort == "inline" else "crossline"
    if free >= int(u8_bytes * 1.1):
        log.warning(
            "Only %s free: caching the %s order only; other axes use strided reads",
            sysinfo.gb(free),
            primary,
        )
        return [primary]
    raise OSError(
        f"Not enough free disk for the cache: need at least {sysinfo.gb(u8_bytes * 1.1)}, "
        f"have {sysinfo.gb(free)} at {directory}"
    )


@dataclass
class _Throughput:
    started: float = field(default_factory=time.perf_counter)
    last_emit: float = 0.0
    last_logged_pct: int = -5


def build_cache(
    path: str | Path,
    directory: str | Path,
    emit: Emit | None = None,
    source: SeismicSource | None = None,
) -> dict:
    """Stream ``path`` into a uint8 memmap cache under ``directory``; return the manifest."""
    path = Path(path)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    emit = emit or (lambda _event: None)
    timings: dict[str, float] = {}
    build_started = time.perf_counter()

    def stage(name: str, message: str) -> float:
        log.info("[%s] %s: %s", path.name, name, message)
        emit({"type": "stage", "stage": name, "message": message})
        return time.perf_counter()

    t = stage("scanning", "reading trace headers")
    own_source = source is None
    source = source or open_source(path)
    try:
        layout = source.layout
        cellmap: np.ndarray | None = None
        if isinstance(source, SegySource) and layout.kind == "3d":
            layout, cellmap = scan_layout_exact(
                source.reader,
                layout,
                progress=lambda f: emit(
                    {"type": "progress", "stage": "scanning", "fraction": f}
                ),
            )
            source.layout = layout
        else:
            layout.provisional = False
        if cellmap is not None:
            np.save(directory / "cellmap.npy", cellmap.astype(np.int64))
        emit(
            {
                "type": "layout",
                "layout": layout.to_dict(),
                "cellmap": str(directory / "cellmap.npy") if cellmap is not None else None,
            }
        )
        timings["scanning"] = time.perf_counter() - t

        t = stage(
            "sampling",
            f"estimating clip percentiles from {config.CACHE_PERCENTILE_TRACES} traces",
        )
        samples = source.sample_traces(config.CACHE_PERCENTILE_TRACES)
        lo, hi = clip_bounds(samples)
        fill = fill_value(lo, hi)
        del samples
        timings["sampling"] = time.perf_counter() - t
        log.info(
            "[%s] clip bounds lo=%.6g hi=%.6g (P%.0f/P%.0f), missing-trace level %d [%.2fs]",
            path.name,
            lo,
            hi,
            config.CLIP_LOW_PERCENTILE,
            config.CLIP_HIGH_PERCENTILE,
            fill,
            timings["sampling"],
        )
        emit({"type": "bounds", "lo": lo, "hi": hi, "fill_u8": fill})

        shape = layout.shape
        # math.prod, not np.prod: the latter overflows int32 on Windows.
        u8_bytes = math.prod(int(s) for s in shape)
        orders = _choose_orders(layout.kind, layout.sort, u8_bytes, directory)

        t = stage(
            "converting",
            f"writing {', '.join(orders)} order(s), {sysinfo.gb(u8_bytes * len(orders))} total",
        )
        memmaps: dict[str, np.memmap] = {}
        for order in orders:
            memmaps[order] = np.memmap(
                directory / f"{order}.u8.partial",
                dtype=np.uint8,
                mode="w+",
                shape=order_shape(order, shape),
            )
        needs_fill = layout.kind == "3d" and (not layout.regular or layout.missing)
        if needs_fill and fill != 0:
            for mm in memmaps.values():
                flat = mm.reshape(-1)
                step = 256 * 1024**2
                for a in range(0, flat.size, step):
                    flat[a : a + step] = fill
        _convert(source, layout, cellmap, memmaps, lo, hi, emit, path.name)
        for mm in memmaps.values():
            _close_memmap(mm)
        memmaps.clear()
        gc.collect()
        timings["converting"] = time.perf_counter() - t

        t = stage("finalizing", "writing geometry and manifest")
        for order in orders:
            os.replace(directory / f"{order}.u8.partial", directory / f"{order}.u8")
        geometry = source.geometry(cellmap)
        save_geometry(directory / "geometry.npz", geometry)
        stat = path.stat()
        timings["finalizing"] = time.perf_counter() - t
        timings["total"] = time.perf_counter() - build_started
        manifest = {
            "version": CACHE_VERSION,
            "source": path.name,
            "source_size": stat.st_size,
            "source_mtime": stat.st_mtime,
            "format": source.format,
            "kind": layout.kind,
            "shape": list(shape),
            "orders": orders,
            "layout": layout.to_dict(),
            "has_cellmap": cellmap is not None,
            "lo": lo,
            "hi": hi,
            "fill_u8": fill,
            "timings": timings,
            "complete": True,
        }
        _write_json_atomic(directory / "manifest.json", manifest)
        log.info(
            "[%s] cache complete in %.1fs (scan %.1fs, sample %.1fs, convert %.1fs) -> %s",
            path.name,
            timings["total"],
            timings["scanning"],
            timings["sampling"],
            timings["converting"],
            directory,
        )
        emit({"type": "done", "manifest": manifest, "directory": str(directory)})
        return manifest
    finally:
        if own_source:
            source.close()


def _block_traces(layout: GridLayout) -> int:
    per_trace = layout.n_samples * 4
    n = max(1, config.CACHE_BLOCK_BYTES // per_trace)
    if layout.kind == "3d" and layout.regular:
        row = layout.n_xl if layout.sort == "inline" else layout.n_il
        n = max(row, (n // row) * row)
    return int(n)


def _convert(
    source: SeismicSource,
    layout: GridLayout,
    cellmap: np.ndarray | None,
    memmaps: dict[str, np.memmap],
    lo: float,
    hi: float,
    emit: Emit,
    name: str,
) -> None:
    n = layout.n_traces
    step = _block_traces(layout)
    bytes_per_trace = layout.n_samples * 4
    blocks: queue.Queue = queue.Queue(maxsize=2)
    stop_flag = threading.Event()

    def reader() -> None:
        try:
            for start in range(0, n, step):
                if stop_flag.is_set():
                    return
                stop = min(n, start + step)
                blocks.put((start, stop, source.read(start, stop)))
            blocks.put(None)
        except BaseException as exc:  # surface read errors to the writer
            blocks.put(exc)

    thread = threading.Thread(target=reader, daemon=True, name=f"cache-read-{name}")
    thread.start()
    meter = _Throughput()
    done = 0
    unflushed = 0
    # Flushing regularly turns dirty cache pages into clean ones the OS can
    # drop, so the build does not pin RAM proportional to the cache size.
    flush_every = 2 * 1024**3
    try:
        while True:
            item = blocks.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            start, stop, block = item
            u8 = normalize_block(block, lo, hi)
            del block
            _scatter(layout, cellmap, memmaps, start, stop, u8)
            unflushed += u8.nbytes * len(memmaps)
            if unflushed >= flush_every:
                for mm in memmaps.values():
                    mm.flush()
                unflushed = 0
            done = stop
            _report(meter, done, n, bytes_per_trace, emit, name)
    finally:
        stop_flag.set()
        thread.join(timeout=5)


def _scatter(
    layout: GridLayout,
    cellmap: np.ndarray | None,
    memmaps: dict[str, np.memmap],
    start: int,
    stop: int,
    u8: np.ndarray,
) -> None:
    """Write a (traces, n_samples) uint8 block into every cached order."""
    if layout.kind != "3d":
        memmaps["section"][:, start:stop] = u8.T
        return
    n_il, n_xl, n_s = layout.shape
    inline = memmaps.get("inline")
    crossline = memmaps.get("crossline")
    time_mm = memmaps.get("time")
    if layout.regular and layout.sort == "inline" and start % n_xl == 0:
        r0, r1 = start // n_xl, stop // n_xl
        blk = u8.reshape(r1 - r0, n_xl, n_s)
        if inline is not None:
            inline[r0:r1] = blk
        if crossline is not None:
            crossline[:, r0:r1, :] = blk.transpose(1, 0, 2)
        if time_mm is not None:
            time_mm[:, r0:r1, :] = blk.transpose(2, 0, 1)
        return
    if layout.regular and layout.sort == "crossline" and start % n_il == 0:
        c0, c1 = start // n_il, stop // n_il
        blk = u8.reshape(c1 - c0, n_il, n_s)
        if crossline is not None:
            crossline[c0:c1] = blk
        if inline is not None:
            inline[:, c0:c1, :] = blk.transpose(1, 0, 2)
        if time_mm is not None:
            time_mm[:, :, c0:c1] = blk.transpose(2, 1, 0)
        return
    cells = cells_for_traces(layout, cellmap, start, stop)
    valid = cells >= 0
    if not np.any(valid):
        return
    il_idx, xl_idx = np.divmod(cells[valid], n_xl)
    values = u8[valid]
    if inline is not None:
        inline[il_idx, xl_idx] = values
    if crossline is not None:
        crossline[xl_idx, il_idx] = values
    if time_mm is not None:
        time_mm[:, il_idx, xl_idx] = values.T


def _report(
    meter: _Throughput, done: int, total: int, bytes_per_trace: int, emit: Emit, name: str
) -> None:
    now = time.perf_counter()
    elapsed = max(1e-6, now - meter.started)
    fraction = done / total if total else 1.0
    mb_per_s = done * bytes_per_trace / elapsed / 1e6
    eta = elapsed * (1 - fraction) / fraction if fraction > 0 else None
    pct = int(fraction * 100)
    if pct >= meter.last_logged_pct + 5 or done == total:
        meter.last_logged_pct = pct - pct % 5
        log.info(
            "[%s] converting %3d%% (%d/%d traces) %.0f MB/s, ETA %s, RSS %s, free disk %s",
            name,
            pct,
            done,
            total,
            mb_per_s,
            f"{eta:.0f}s" if eta is not None else "?",
            sysinfo.gb(sysinfo.process_rss()),
            sysinfo.gb(sysinfo.free_disk(config.CACHE_DIR)),
        )
    if now - meter.last_emit >= 0.25 or done == total:
        meter.last_emit = now
        emit(
            {
                "type": "progress",
                "stage": "converting",
                "fraction": fraction,
                "mb_per_s": mb_per_s,
                "eta_s": eta,
            }
        )


# ---- background manager -----------------------------------------------------


def _process_main(path: str, directory: str, events) -> None:
    """Child-process entry point: build one cache, stream events to the parent."""
    try:
        build_cache(path, directory, events.put)
    except BaseException as exc:
        log.exception("Cache build failed for %s", path)
        events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        events.put({"type": "exit"})


@dataclass
class BuildState:
    """What the API knows about one large file's cache."""

    name: str
    path: Path
    directory: Path
    stage: str = "queued"
    message: str = "waiting for another file's cache to finish"
    fraction: float = 0.0
    mb_per_s: float | None = None
    eta_s: float | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    layout: GridLayout | None = None
    cellmap_path: str | None = None
    lo: float | None = None
    hi: float | None = None
    fill_u8: int = 0
    manifest: dict | None = None
    version: int = 0  # bumped on every change so volumes can be refreshed

    @property
    def ready(self) -> bool:
        return self.manifest is not None

    def to_json(self) -> dict:
        percent = 100.0 if self.ready else round(100.0 * self._overall(), 1)
        elapsed = None
        if self.started_at is not None:
            elapsed = (self.finished_at or time.time()) - self.started_at
        return {
            "stage": "ready" if self.ready else self.stage,
            "message": self.message,
            "percent": percent,
            "stage_fraction": round(self.fraction, 4),
            "mb_per_s": round(self.mb_per_s, 1) if self.mb_per_s else None,
            "eta_s": round(self.eta_s, 1) if self.eta_s is not None else None,
            "elapsed_s": round(elapsed, 1) if elapsed is not None else None,
            "error": self.error,
            "ready": self.ready,
            "cache_dir": str(self.directory),
        }

    def _overall(self) -> float:
        # Scanning and sampling are short next to converting.
        weights = {"queued": (0.0, 0.0), "scanning": (0.0, 0.05), "sampling": (0.05, 0.03),
                   "converting": (0.08, 0.9), "finalizing": (0.98, 0.02)}
        base, span = weights.get(self.stage, (0.0, 0.0))
        return min(1.0, base + span * self.fraction)


class CacheManager:
    """Queue of cache builds, one at a time, open file first."""

    def __init__(self, root: Path | None = None, use_process: bool = True):
        self.root = Path(root or config.CACHE_DIR)
        self.use_process = use_process
        self._states: dict[str, BuildState] = {}
        self._pending: list[str] = []
        self._cv = threading.Condition()
        self._worker: threading.Thread | None = None
        self._process = None
        self._closed = False
        self.listeners: list[Callable[[BuildState], None]] = []

    # -- public API --

    def state(self, name: str) -> BuildState | None:
        with self._cv:
            return self._states.get(name)

    def ensure(self, name: str, path: Path, priority: bool = False, retry: bool = False) -> BuildState:
        """Return the file's state, queueing a build when no valid cache exists."""
        path = Path(path)
        directory = cache_dir_for(path, self.root)
        with self._cv:
            state = self._states.get(name)
            if state is not None and state.directory != directory:
                log.info("%s changed on disk; its cache will be rebuilt", name)
                if name in self._pending:
                    self._pending.remove(name)
                state = None
            if state is None:
                state = BuildState(name=name, path=path, directory=directory)
                manifest = read_manifest(directory)
                if manifest is not None:
                    self._mark_ready(state, manifest)
                    log.info("%s: cache hit at %s", name, directory)
                    self._states[name] = state
                    return state
                log.info(
                    "%s: no valid cache at %s (%s); queueing build",
                    name,
                    directory,
                    "partial build found" if directory.exists() else "first use",
                )
                self._states[name] = state
                self._pending.append(name)
            elif state.error and retry:
                log.info("%s: retrying failed cache build (%s)", name, state.error)
                state.error = None
                state.stage = "queued"
                state.message = "retrying"
                state.version += 1
                self._pending.append(name)
            if priority and name in self._pending and self._pending[0] != name:
                self._pending.remove(name)
                self._pending.insert(0, name)
                log.info("%s moved to the front of the cache queue", name)
            self._start_worker_locked()
            self._cv.notify_all()
            return state

    def shutdown(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
            process = self._process
        if process is not None and process.is_alive():
            log.info("Stopping in-flight cache build")
            process.terminate()

    # -- internals --

    def _mark_ready(self, state: BuildState, manifest: dict) -> None:
        state.manifest = manifest
        state.layout = GridLayout.from_dict(manifest["layout"])
        state.lo, state.hi, state.fill_u8 = manifest["lo"], manifest["hi"], manifest["fill_u8"]
        state.stage = "ready"
        state.message = "cache ready"
        state.fraction = 1.0
        state.version += 1

    def _start_worker_locked(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, daemon=True, name="cache-manager")
            self._worker.start()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                name = self._pending.pop(0)
                state = self._states[name]
            self._build(state)

    def _notify(self, state: BuildState) -> None:
        state.version += 1
        for listener in list(self.listeners):
            try:
                listener(state)
            except Exception:
                log.exception("cache listener failed")

    def _apply(self, state: BuildState, event: dict) -> None:
        kind = event.get("type")
        if kind == "stage":
            state.stage = event["stage"]
            state.message = event.get("message", "")
            state.fraction = 0.0
        elif kind == "progress":
            state.stage = event.get("stage", state.stage)
            state.fraction = float(event.get("fraction", 0.0))
            state.mb_per_s = event.get("mb_per_s", state.mb_per_s)
            state.eta_s = event.get("eta_s")
        elif kind == "layout":
            state.layout = GridLayout.from_dict(event["layout"])
            state.cellmap_path = event.get("cellmap")
        elif kind == "bounds":
            state.lo, state.hi, state.fill_u8 = event["lo"], event["hi"], event["fill_u8"]
        elif kind == "done":
            manifest = event["manifest"]
            self._mark_ready(state, manifest)
            state.finished_at = time.time()
            remove_stale_caches(self.root, state.name, state.directory)
        elif kind == "error":
            state.error = event["message"]
            state.stage = "error"
            state.message = event["message"]
            state.finished_at = time.time()
        self._notify(state)

    def _build(self, state: BuildState) -> None:
        state.started_at = time.time()
        state.finished_at = None
        state.stage = "scanning"
        state.message = "starting"
        self._notify(state)
        log.info("Starting cache build for %s -> %s", state.name, state.directory)
        if not self.use_process:
            try:
                build_cache(state.path, state.directory, lambda ev: self._apply(state, ev))
            except Exception as exc:
                log.exception("Cache build failed for %s", state.name)
                self._apply(state, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            return
        ctx = mp.get_context("spawn")
        events = ctx.Queue()
        process = ctx.Process(
            target=_process_main,
            args=(str(state.path), str(state.directory), events),
            daemon=True,
            name=f"cache-build-{state.name}",
        )
        with self._cv:
            self._process = process
        process.start()
        finished = False
        while not finished:
            try:
                event = events.get(timeout=1.0)
            except queue.Empty:
                if not process.is_alive():
                    break
                continue
            if event.get("type") == "exit":
                finished = True
                continue
            self._apply(state, event)
        process.join(timeout=10)
        with self._cv:
            self._process = None
        if not state.ready and state.error is None:
            code = process.exitcode
            self._apply(
                state,
                {"type": "error", "message": f"cache build process exited with code {code}"},
            )
