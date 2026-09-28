"""Fast, header-layout-agnostic SEG-Y geometry for files of any size.

segyio's structured mode (``segyio.open(iline=..., xline=...)``) reads
every trace header before returning and only accepts a complete,
perfectly sorted grid. On multi-GB volumes that takes minutes and it
rejects irregular surveys outright. This module instead:

- reads trace headers and samples straight from a ``np.memmap`` of the
  file (fixed-length traces, which is what SEG-Y specifies), decoding
  IBM floats and the other sample formats with vectorized NumPy on a
  thread pool - roughly 20x faster than ``segyio``'s per-trace path;
- detects the inline/crossline header bytes and the sort order
  (inline-major or crossline-major) from a few thousand sampled headers;
- scans the chosen header pair for every trace once (in the background)
  to build an exact ``trace -> grid cell`` map, so surveys with missing
  or duplicated traces still land on a regular cube.

Anything the fast path cannot handle (variable-length traces, unknown
sample formats) falls back to segyio's own readers.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import segyio

from .geometry import (
    SectionGeometry,
    _coordinate_scalar,
    _maybe_decode_ieee,
    _read_dt_ms,
    extract_2d_geometry,
)
from .logutil import get_logger

log = get_logger("segy_geometry")

# Inline/crossline header byte pairs tried, in order. 189/193 is the
# SEG-Y rev1 standard; the rest are common vendor layouts. 181/185
# (CDP X/Y) only yields a grid for axis-aligned surveys.
GRID_BYTE_CANDIDATES: list[tuple[int, int]] = [
    (189, 193),
    (17, 25),
    (9, 13),
    (9, 21),
    (5, 21),
    (21, 25),
    (13, 17),
    (181, 185),
]

# SEG-Y sample format code -> (bytes per sample, numpy dtype or "ibm").
SAMPLE_FORMATS: dict[int, tuple[int, str]] = {
    1: (4, "ibm"),
    2: (4, "i4"),
    3: (2, "i2"),
    5: (4, "f4"),
    6: (8, "f8"),
    8: (1, "i1"),
    9: (8, "i8"),
    10: (4, "u4"),
    11: (2, "u2"),
    12: (8, "u8"),
    16: (1, "u1"),
}

_HEAD_TRACES = 5000
_SPACED_TRACES = 257
# A detected grid must cover the traces without being implausibly sparse.
_MIN_FILL_RATIO = 0.25
_DECODE_THREADS = max(1, min(12, (os.cpu_count() or 4) - 2))

with np.errstate(over="ignore", under="ignore"):
    _IBM_SCALE = np.ldexp(
        np.float32(1.0), 4 * (np.arange(128) - 64) - 24
    ).astype(np.float32)


def _header_width(byte: int) -> int:
    """Width in bytes of the SEG-Y rev1 trace header field starting at ``byte``."""
    if 29 <= byte <= 36 or 69 <= byte <= 72 or 89 <= byte <= 180:
        return 2
    return 4


def detect_endian(path: str | Path) -> str:
    """'big' (the SEG-Y standard) unless the binary header only parses little-endian."""
    with open(path, "rb") as fh:
        fh.seek(3224)
        raw = fh.read(2)
    if len(raw) < 2:
        return "big"
    if int.from_bytes(raw, "big") in SAMPLE_FORMATS:
        return "big"
    if int.from_bytes(raw, "little") in SAMPLE_FORMATS:
        return "little"
    return "big"


def _ibm_to_float32(words: np.ndarray) -> np.ndarray:
    """IBM System/360 floats (as native uint32) -> IEEE float32, bit-exact with segyio."""
    out = (words & 0x00FFFFFF).astype(np.float32)
    out *= _IBM_SCALE[(words >> 24) & 0x7F]
    out.view(np.uint32)[...] |= words & 0x80000000
    return out


def _decode_rows(raw: np.ndarray, code: int, endian_char: str, n_samples: int) -> np.ndarray:
    """Decode a (rows, n_samples * bps) uint8 block into float32 samples."""
    raw = np.ascontiguousarray(raw)
    bps, kind = SAMPLE_FORMATS[code]
    if kind == "ibm":
        words = raw.view(f"{endian_char}u4").astype(np.uint32)
        out = _ibm_to_float32(words)
    else:
        out = raw.view(f"{endian_char}{kind}").astype(np.float32)
    return out.reshape(raw.shape[0], n_samples)


@dataclass
class RawLayout:
    """Byte layout of a fixed-trace-length SEG-Y file."""

    data_offset: int
    trace_bytes: int
    n_traces: int
    n_samples: int
    format_code: int
    endian: str


class TraceReader:
    """Thread-safe trace/header access for one SEG-Y file.

    Uses a read-only memmap plus vectorized decoding when the file has
    fixed-length traces of a known sample format; otherwise delegates to
    segyio under a lock (segyio handles are not thread-safe).
    """

    def __init__(self, path: str | Path, threads: int | None = None):
        self.path = Path(path)
        self.endian = detect_endian(self.path)
        self._lock = threading.Lock()
        self.segy = segyio.open(str(self.path), ignore_geometry=True, endian=self.endian)
        self.n_traces = int(self.segy.tracecount)
        self.n_samples = int(len(self.segy.samples))
        self.format_code = int(self.segy.bin[segyio.BinField.Format])
        self.raw: RawLayout | None = None
        self._mm: np.ndarray | None = None
        self._threads = threads or _DECODE_THREADS
        self._pool: ThreadPoolExecutor | None = None
        self._init_raw()

    def _init_raw(self) -> None:
        if self.format_code not in SAMPLE_FORMATS:
            log.info(
                "%s: sample format %d has no fast decoder; using segyio reads",
                self.path.name,
                self.format_code,
            )
            return
        ext = int(self.segy.ext_headers or 0)
        if ext < 0:
            return
        bps = SAMPLE_FORMATS[self.format_code][0]
        offset = 3600 + 3200 * ext
        trace_bytes = 240 + self.n_samples * bps
        size = self.path.stat().st_size
        if size != offset + self.n_traces * trace_bytes:
            log.info(
                "%s: file size does not match fixed-length traces; using segyio reads",
                self.path.name,
            )
            return
        self.raw = RawLayout(
            offset, trace_bytes, self.n_traces, self.n_samples, self.format_code, self.endian
        )
        self._mm = np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            offset=offset,
            shape=(self.n_traces, trace_bytes),
        )

    @property
    def fast(self) -> bool:
        return self._mm is not None

    @property
    def _endian_char(self) -> str:
        return ">" if self.endian == "big" else "<"

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        self._mm = None
        with self._lock:
            try:
                self.segy.close()
            except Exception:
                pass

    def __enter__(self) -> "TraceReader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ---- headers -------------------------------------------------------

    def _read_rows(self, start: int, stop: int) -> np.ndarray:
        """Raw (n, trace_bytes) uint8 rows via a plain sequential file read.

        Bulk reads avoid the memmap so the pages land in the OS file cache
        rather than this process's working set.
        """
        assert self.raw is not None
        n = stop - start
        with open(self.path, "rb") as fh:
            fh.seek(self.raw.data_offset + start * self.raw.trace_bytes)
            buf = np.fromfile(fh, dtype=np.uint8, count=n * self.raw.trace_bytes)
        return buf.reshape(n, self.raw.trace_bytes)

    def header_fields(
        self, bytes_: list[int], start: int = 0, stop: int | None = None
    ) -> list[np.ndarray]:
        """Several trace header fields for traces [start, stop), read in one pass."""
        stop = self.n_traces if stop is None else min(stop, self.n_traces)
        if self._mm is None:
            return [self.header_field(byte, start, stop) for byte in bytes_]
        rows = self._read_rows(start, stop) if stop - start > 4096 else self._mm[start:stop]
        out = []
        for byte in bytes_:
            width = _header_width(byte)
            cols = rows[:, byte - 1 : byte - 1 + width]
            dtype = f"{self._endian_char}i{width}"
            out.append(np.ascontiguousarray(cols).view(dtype).ravel().astype(np.int64))
        return out

    def header_field(self, byte: int, start: int = 0, stop: int | None = None) -> np.ndarray:
        """One trace header field for traces [start, stop) as int64."""
        stop = self.n_traces if stop is None else min(stop, self.n_traces)
        if self._mm is not None:
            return self.header_fields([byte], start, stop)[0]
        with self._lock:
            return np.asarray(self.segy.attributes(byte)[start:stop], dtype=np.int64)

    def header_field_at(self, byte: int, indices: np.ndarray) -> np.ndarray:
        """One trace header field for arbitrary trace indices as int64."""
        indices = np.asarray(indices, dtype=np.int64)
        if self._mm is not None:
            width = _header_width(byte)
            cols = self._mm[indices, byte - 1 : byte - 1 + width]
            dtype = f"{self._endian_char}i{width}"
            return np.ascontiguousarray(cols).view(dtype).ravel().astype(np.int64)
        with self._lock:
            return np.asarray(
                [self.segy.header[int(i)][byte] for i in indices], dtype=np.int64
            )

    # ---- samples -------------------------------------------------------

    def _decode(self, rows: np.ndarray) -> np.ndarray:
        assert self.raw is not None
        return _decode_rows(rows, self.format_code, self._endian_char, self.n_samples)

    def read(self, start: int, stop: int) -> np.ndarray:
        """Traces [start, stop) as a (n, n_samples) float32 array."""
        stop = min(stop, self.n_traces)
        if stop <= start:
            return np.zeros((0, self.n_samples), dtype=np.float32)
        if self._mm is None:
            with self._lock:
                return np.asarray(self.segy.trace.raw[start:stop], dtype=np.float32)
        n = stop - start
        if n < 2048 or self._threads <= 1:
            return self._decode(self._mm[start:stop, 240:])
        rows = self._read_rows(start, stop)
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self._threads, thread_name_prefix="segy-decode")
        out = np.empty((n, self.n_samples), dtype=np.float32)
        bounds = np.linspace(0, n, self._threads + 1).astype(np.int64)

        def work(k: int) -> None:
            a, b = int(bounds[k]), int(bounds[k + 1])
            if b > a:
                out[a:b] = self._decode(rows[a:b, 240:])

        list(self._pool.map(work, range(self._threads)))
        return out

    def read_indices(self, indices: np.ndarray) -> np.ndarray:
        """Arbitrary traces as a (len(indices), n_samples) float32 array."""
        indices = np.asarray(indices, dtype=np.int64)
        if self._mm is not None:
            return self._decode(self._mm[indices, 240:])
        with self._lock:
            return np.stack(
                [np.asarray(self.segy.trace[int(i)], dtype=np.float32) for i in indices]
            ) if len(indices) else np.zeros((0, self.n_samples), dtype=np.float32)


# ---- grid layout -------------------------------------------------------


@dataclass
class GridLayout:
    """Where every trace of a file sits: a 3D grid or a 2D line."""

    kind: str  # "3d" or "2d"
    n_traces: int
    n_samples: int
    il_byte: int | None = None
    xl_byte: int | None = None
    # "inline": crossline varies fastest in file order (segyio's inline
    # sorting); "crossline": inline varies fastest.
    sort: str = "inline"
    il_min: int = 0
    il_step: int = 1
    n_il: int = 1
    xl_min: int = 0
    xl_step: int = 1
    n_xl: int = 1
    # Every trace sits exactly at grid position = file order.
    regular: bool = False
    # Sampled headers only; the exact scan has not run yet.
    provisional: bool = True
    missing: int = 0
    duplicates: int = 0
    invalid: int = 0

    @property
    def shape(self) -> tuple[int, ...]:
        if self.kind == "3d":
            return (self.n_il, self.n_xl, self.n_samples)
        return (self.n_samples, self.n_traces)

    @property
    def n_cells(self) -> int:
        return self.n_il * self.n_xl

    def ilines(self) -> np.ndarray:
        return self.il_min + self.il_step * np.arange(self.n_il, dtype=np.int64)

    def xlines(self) -> np.ndarray:
        return self.xl_min + self.xl_step * np.arange(self.n_xl, dtype=np.int64)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GridLayout":
        fields = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**fields)

    def describe(self) -> str:
        if self.kind != "3d":
            return f"2D line: {self.n_traces} traces x {self.n_samples} samples"
        return (
            f"3D grid {self.n_il} IL x {self.n_xl} XL x {self.n_samples} samples "
            f"(bytes {self.il_byte}/{self.xl_byte}, {self.sort}-sorted, "
            f"IL {self.il_min}+{self.il_step}k, XL {self.xl_min}+{self.xl_step}k, "
            f"{'regular' if self.regular else 'irregular'}"
            f"{', provisional' if self.provisional else ''}"
            f"{f', {self.missing} missing' if self.missing else ''}"
            f"{f', {self.duplicates} duplicate' if self.duplicates else ''})"
        )


def _gcd_step(values: np.ndarray) -> int:
    unique = np.unique(values)
    if unique.size < 2:
        return 0
    diffs = np.diff(unique)
    return int(np.gcd.reduce(diffs))


def _read_override(path: Path) -> dict | None:
    """Per-file header-byte override: ``data/<file>.segy.json`` or SEISMIC_SEGY_BYTES."""
    for sidecar in (
        path.with_name(path.name + ".segy.json"),
        path.with_name(path.stem + ".segy.json"),
    ):
        if sidecar.is_file():
            try:
                data = json.loads(sidecar.read_text())
                log.info("%s: using header override from %s: %s", path.name, sidecar.name, data)
                return data
            except (OSError, ValueError) as exc:
                log.warning("Ignoring unreadable override %s: %s", sidecar, exc)
    env = os.environ.get("SEISMIC_SEGY_BYTES", "").strip()
    if env:
        try:
            il_byte, xl_byte = (int(v) for v in env.split(","))
            return {"iline_byte": il_byte, "xline_byte": xl_byte}
        except ValueError:
            log.warning("Ignoring SEISMIC_SEGY_BYTES=%r (expected 'iline,xline')", env)
    return None


def _grid_from_values(
    il: np.ndarray,
    xl: np.ndarray,
    head_il: np.ndarray,
    head_xl: np.ndarray,
    n_traces: int,
    sort_hint: str | None = None,
) -> dict | None:
    """Estimate a grid from inline/crossline values; None when they do not form one."""
    if np.all(il == il[0]) or np.all(xl == xl[0]):
        return None
    il_step, xl_step = _gcd_step(il), _gcd_step(xl)
    if il_step <= 0 or xl_step <= 0:
        return None
    il_min, il_max = int(il.min()), int(il.max())
    xl_min, xl_max = int(xl.min()), int(xl.max())
    n_il = (il_max - il_min) // il_step + 1
    n_xl = (xl_max - xl_min) // xl_step + 1
    cells = n_il * n_xl
    if n_il < 2 or n_xl < 2 or cells < n_traces or cells * _MIN_FILL_RATIO > n_traces:
        return None
    if sort_hint in ("inline", "crossline"):
        sort = sort_hint
    else:
        il_changes = int(np.count_nonzero(np.diff(head_il)))
        xl_changes = int(np.count_nonzero(np.diff(head_xl)))
        sort = "inline" if xl_changes >= il_changes else "crossline"
    return {
        "sort": sort,
        "il_min": il_min,
        "il_step": il_step,
        "n_il": n_il,
        "xl_min": xl_min,
        "xl_step": xl_step,
        "n_xl": n_xl,
    }


def _expected_cells(layout: GridLayout, traces: np.ndarray) -> np.ndarray:
    """Grid cell (il_idx * n_xl + xl_idx) of each trace for a regular file."""
    if layout.sort == "inline":
        return traces
    xl_idx, il_idx = np.divmod(traces, layout.n_il)
    return il_idx * layout.n_xl + xl_idx


def _cells_of(layout: GridLayout, il: np.ndarray, xl: np.ndarray) -> np.ndarray:
    il_off = il - layout.il_min
    xl_off = xl - layout.xl_min
    il_idx, il_rem = np.divmod(il_off, layout.il_step)
    xl_idx, xl_rem = np.divmod(xl_off, layout.xl_step)
    valid = (
        (il_rem == 0)
        & (xl_rem == 0)
        & (il_idx >= 0)
        & (il_idx < layout.n_il)
        & (xl_idx >= 0)
        & (xl_idx < layout.n_xl)
    )
    cells = np.where(valid, il_idx * layout.n_xl + xl_idx, -1)
    return cells.astype(np.int64)


def detect_layout_fast(reader: TraceReader, path: Path | None = None) -> GridLayout:
    """Provisional layout from ~5000 leading plus ~257 spaced trace headers."""
    started = time.perf_counter()
    path = path or reader.path
    n = reader.n_traces
    base = GridLayout(kind="2d", n_traces=n, n_samples=reader.n_samples)
    if n < 4:
        return base
    override = _read_override(path)
    if override and str(override.get("kind", "")).lower() == "2d":
        return base
    candidates = GRID_BYTE_CANDIDATES
    sort_hint = None
    if override and "iline_byte" in override and "xline_byte" in override:
        candidates = [(int(override["iline_byte"]), int(override["xline_byte"]))]
        sort_hint = override.get("sort")

    head_n = min(n, _HEAD_TRACES)
    spaced = np.unique(np.linspace(0, n - 1, _SPACED_TRACES).astype(np.int64))
    spaced = spaced[spaced >= head_n]
    sample_idx = np.concatenate([np.arange(head_n, dtype=np.int64), spaced])

    best: tuple[float, GridLayout] | None = None
    tried: list[str] = []
    for il_byte, xl_byte in candidates:
        try:
            head_il = reader.header_field(il_byte, 0, head_n)
            head_xl = reader.header_field(xl_byte, 0, head_n)
            il = np.concatenate([head_il, reader.header_field_at(il_byte, spaced)])
            xl = np.concatenate([head_xl, reader.header_field_at(xl_byte, spaced)])
        except Exception as exc:  # unreadable field layout
            tried.append(f"{il_byte}/{xl_byte}: {exc}")
            continue
        grid = _grid_from_values(il, xl, head_il, head_xl, n, sort_hint)
        if grid is None:
            tried.append(f"{il_byte}/{xl_byte}: no grid")
            continue
        layout = GridLayout(
            kind="3d",
            n_traces=n,
            n_samples=reader.n_samples,
            il_byte=il_byte,
            xl_byte=xl_byte,
            **grid,
        )
        if layout.n_cells == n:
            expected = _expected_cells(layout, sample_idx)
            layout.regular = bool(np.array_equal(_cells_of(layout, il, xl), expected))
        fill = n / layout.n_cells
        score = (2.0 if layout.regular else 0.0) + fill
        tried.append(
            f"{il_byte}/{xl_byte}: {layout.n_il}x{layout.n_xl} fill {fill:.2f}"
            f"{' regular' if layout.regular else ''}"
        )
        if best is None or score > best[0]:
            best = (score, layout)
        if layout.regular:
            break
    elapsed = time.perf_counter() - started
    if best is None:
        log.info(
            "%s: no 3D grid in headers (%s); treating as a 2D line [%.2fs]",
            path.name,
            "; ".join(tried),
            elapsed,
        )
        return base
    layout = best[1]
    log.info(
        "%s: detected %s from %d sampled headers [%.2fs]; candidates: %s",
        path.name,
        layout.describe(),
        len(sample_idx),
        elapsed,
        "; ".join(tried),
    )
    return layout


def scan_layout_exact(
    reader: TraceReader,
    layout: GridLayout,
    progress: Callable[[float], None] | None = None,
    block: int = 1_000_000,
) -> tuple[GridLayout, np.ndarray | None]:
    """Read the grid header pair for every trace; return (layout, cellmap).

    The grid bounds and steps are recomputed from all traces (the fast
    estimate can miss extremes). ``cellmap`` maps trace -> grid cell
    (-1 for traces off the grid) and is None when the file is regular,
    i.e. traces are exactly in grid order.
    """
    if layout.kind != "3d" or layout.il_byte is None or layout.xl_byte is None:
        return layout, None
    started = time.perf_counter()
    n = reader.n_traces
    il = np.empty(n, dtype=np.int64)
    xl = np.empty(n, dtype=np.int64)
    if reader.raw is not None:
        # Keep each sequential read around 256 MB whatever the trace length.
        block = max(1024, (256 * 1024**2) // reader.raw.trace_bytes)
    for start in range(0, n, block):
        stop = min(n, start + block)
        il[start:stop], xl[start:stop] = reader.header_fields(
            [layout.il_byte, layout.xl_byte], start, stop
        )
        if progress is not None:
            progress(stop / n)

    head_n = min(n, _HEAD_TRACES)
    grid = _grid_from_values(il, xl, il[:head_n], xl[:head_n], n, layout.sort)
    exact = GridLayout(
        kind="3d",
        n_traces=n,
        n_samples=layout.n_samples,
        il_byte=layout.il_byte,
        xl_byte=layout.xl_byte,
        provisional=False,
    )
    if grid is None:
        # Headers lied under full inspection (e.g. many duplicates);
        # keep the sampled grid and drop what does not fit it.
        for key in ("sort", "il_min", "il_step", "n_il", "xl_min", "xl_step", "n_xl"):
            setattr(exact, key, getattr(layout, key))
    else:
        for key, value in grid.items():
            setattr(exact, key, value)

    cells = _cells_of(exact, il, xl)
    valid = cells >= 0
    exact.invalid = int(n - np.count_nonzero(valid))
    if exact.n_cells == n and exact.invalid == 0:
        expected = _expected_cells(exact, np.arange(n, dtype=np.int64))
        exact.regular = bool(np.array_equal(cells, expected))
    counts = np.bincount(cells[valid], minlength=exact.n_cells)
    exact.missing = int(np.count_nonzero(counts == 0))
    exact.duplicates = int(np.sum(np.maximum(counts - 1, 0)))
    log.info(
        "%s: exact header scan of %d traces -> %s, %d off-grid [%.2fs]",
        reader.path.name,
        n,
        exact.describe(),
        exact.invalid,
        time.perf_counter() - started,
    )
    if exact.duplicates:
        log.warning(
            "%s: %d traces share a grid cell with another trace; the later trace wins",
            reader.path.name,
            exact.duplicates,
        )
    return exact, (None if exact.regular else cells)


def cells_for_traces(layout: GridLayout, cellmap: np.ndarray | None, start: int, stop: int) -> np.ndarray:
    """Grid cells of traces [start, stop) (regular files need no map)."""
    if cellmap is not None:
        return cellmap[start:stop]
    return _expected_cells(layout, np.arange(start, stop, dtype=np.int64))


def _sampled_world_coords(reader: TraceReader, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = reader.header_field_at(int(segyio.TraceField.CDP_X), indices).astype(np.float64)
    y = reader.header_field_at(int(segyio.TraceField.CDP_Y), indices).astype(np.float64)
    if not (np.any(x != 0) or np.any(y != 0)):
        x = reader.header_field_at(int(segyio.TraceField.SourceX), indices).astype(np.float64)
        y = reader.header_field_at(int(segyio.TraceField.SourceY), indices).astype(np.float64)
    x, y = _maybe_decode_ieee(x), _maybe_decode_ieee(y)
    scalar = _coordinate_scalar(
        int(reader.header_field_at(int(segyio.TraceField.SourceGroupScalar), indices[:1])[0])
    )
    return x * scalar, y * scalar


def _bin_spacing(
    reader: TraceReader, layout: GridLayout, cellmap: np.ndarray | None
) -> tuple[float, float]:
    """Inline/crossline bin size from a least-squares fit of sampled coordinates."""
    n = reader.n_traces
    idx = np.unique(np.linspace(0, n - 1, min(n, 2049)).astype(np.int64))
    cells = cellmap[idx] if cellmap is not None else _expected_cells(layout, idx)
    keep = cells >= 0
    idx, cells = idx[keep], cells[keep]
    if idx.size < 3:
        return 25.0, 25.0
    try:
        x, y = _sampled_world_coords(reader, idx)
    except Exception:
        return 25.0, 25.0
    if not (np.any(x != 0) or np.any(y != 0)):
        return 25.0, 25.0
    il_idx, xl_idx = np.divmod(cells, layout.n_xl)
    design = np.column_stack([np.ones_like(il_idx), il_idx, xl_idx]).astype(np.float64)
    try:
        coef_x, *_ = np.linalg.lstsq(design, x, rcond=None)
        coef_y, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return 25.0, 25.0
    il_sp = float(math.hypot(coef_x[1], coef_y[1]))
    xl_sp = float(math.hypot(coef_x[2], coef_y[2]))
    il_sp = il_sp if 0.01 < il_sp < 1e5 else 25.0
    xl_sp = xl_sp if 0.01 < xl_sp < 1e5 else 25.0
    return il_sp, xl_sp


def build_geometry(
    reader: TraceReader, layout: GridLayout, cellmap: np.ndarray | None = None
) -> SectionGeometry:
    """SectionGeometry for a detected layout without a full segyio index scan."""
    with reader._lock:
        dt_ms = _read_dt_ms(reader.segy)
        t0_ms = float(reader.segy.header[0][segyio.TraceField.DelayRecordingTime])
    if layout.kind != "3d":
        with reader._lock:
            return extract_2d_geometry(reader.segy)
    il_sp, xl_sp = _bin_spacing(reader, layout, cellmap)
    return SectionGeometry(
        kind="3d",
        n_traces=layout.n_traces,
        n_samples=layout.n_samples,
        dt_ms=dt_ms,
        t0_ms=t0_ms,
        ilines=layout.ilines(),
        xlines=layout.xlines(),
        iline_spacing_m=il_sp,
        xline_spacing_m=xl_sp,
    )


def listing_geometry(reader: TraceReader, layout: GridLayout) -> SectionGeometry:
    """Cheap geometry for the dropdown: axes and dt only, no coordinate reads."""
    with reader._lock:
        dt_ms = _read_dt_ms(reader.segy)
    if layout.kind == "3d":
        return SectionGeometry(
            kind="3d",
            n_traces=layout.n_traces,
            n_samples=layout.n_samples,
            dt_ms=dt_ms,
            t0_ms=0.0,
            ilines=layout.ilines(),
            xlines=layout.xlines(),
        )
    return SectionGeometry(
        kind="2d",
        n_traces=layout.n_traces,
        n_samples=layout.n_samples,
        dt_ms=dt_ms,
        t0_ms=0.0,
    )
