"""SEG-Y geometry extraction: the physical axes of a seismic file.

A .sgy file is not a photograph - its samples live on physical axes
(trace position along a 2D line, or inline/crossline for a 3D volume,
plus time/depth down the trace). This module reads that geometry from
the SEG-Y headers once, at load time, so every later stage (display,
point picking, mask stitching, ParaView export) can agree on where each
pixel actually sits in the ground.

Conventions used throughout the app after loading:
- 2D section arrays are (n_samples, n_traces): row = time sample
  (increasing downward), column = trace along the line.
- 3D volume arrays are (n_ilines, n_xlines, n_samples) as returned by
  segyio.tools.cube.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import segyio

# Plausible seismic sample intervals, in microseconds. Some files carry a
# bogus value in the binary header (e.g. data/1.sgy says 4 us) while the
# trace headers hold the real one (4000 us = 4 ms), so every candidate is
# sanity-checked against this range before use.
_MIN_DT_US = 100.0
_MAX_DT_US = 32000.0
_DEFAULT_DT_US = 4000.0


@dataclass
class SectionGeometry:
    """Physical geometry of one SEG-Y file (2D line or 3D volume)."""

    kind: str  # "2d" or "3d"
    n_traces: int  # traces along the line (2D) / total traces (3D)
    n_samples: int
    dt_ms: float  # sample interval, milliseconds
    t0_ms: float  # recording delay of the first sample, milliseconds

    # --- 2D line fields (None for 3D volumes) ---
    cdp: np.ndarray | None = None  # per-trace CDP number
    world_x: np.ndarray | None = None  # per-trace world X (scalar applied)
    world_y: np.ndarray | None = None  # per-trace world Y (scalar applied)
    distance_m: np.ndarray | None = None  # cumulative distance along the line
    trace_spacing_m: float = 25.0  # median spacing between adjacent traces

    # --- 3D volume fields (None for 2D lines) ---
    ilines: np.ndarray | None = None
    xlines: np.ndarray | None = None
    iline_spacing_m: float | None = None
    xline_spacing_m: float | None = None

    # ---- axis helpers ------------------------------------------------

    def sample_times_ms(self) -> np.ndarray:
        """Time (ms) of every sample down the trace."""
        return self.t0_ms + np.arange(self.n_samples) * self.dt_ms

    def sample_to_time_ms(self, sample_idx: float) -> float:
        return self.t0_ms + sample_idx * self.dt_ms

    def time_ms_to_sample(self, time_ms: float) -> int:
        idx = round((time_ms - self.t0_ms) / self.dt_ms)
        return int(np.clip(idx, 0, self.n_samples - 1))

    def trace_to_cdp(self, trace_idx: int) -> int | None:
        """CDP number of a trace (2D lines), or None when unavailable."""
        if self.cdp is None or len(self.cdp) == 0:
            return None
        trace_idx = int(np.clip(trace_idx, 0, len(self.cdp) - 1))
        return int(self.cdp[trace_idx])

    def describe_point(self, trace_idx: int, sample_idx: int) -> str:
        """Human-readable physical location of an array index pair."""
        time_ms = self.sample_to_time_ms(sample_idx)
        cdp = self.trace_to_cdp(trace_idx)
        if cdp is not None:
            return f"CDP {cdp} @ {time_ms:.0f} ms"
        return f"trace {trace_idx} @ {time_ms:.0f} ms"


def _coordinate_scalar(raw: int) -> float:
    """SEG-Y coordinate scalar: positive multiplies, negative divides."""
    if raw > 0:
        return float(raw)
    if raw < 0:
        return 1.0 / abs(raw)
    return 1.0


def _read_dt_ms(f: segyio.SegyFile) -> float:
    """Sample interval in ms, cross-checking trace vs binary header.

    The trace header value is preferred; the binary header is the
    fallback; either is rejected if outside the plausible range.
    """
    candidates = [
        float(f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL]),
        float(f.bin[segyio.BinField.Interval]),
    ]
    for dt_us in candidates:
        if _MIN_DT_US <= dt_us <= _MAX_DT_US:
            return dt_us / 1000.0
    return _DEFAULT_DT_US / 1000.0


def _read_world_coords(f: segyio.SegyFile) -> tuple[np.ndarray, np.ndarray]:
    """Per-trace world (x, y), preferring CDP_X/Y over SourceX/Y."""
    cdp_x = np.asarray(f.attributes(segyio.TraceField.CDP_X)[:], dtype=np.float64)
    cdp_y = np.asarray(f.attributes(segyio.TraceField.CDP_Y)[:], dtype=np.float64)
    if np.any(cdp_x != 0) or np.any(cdp_y != 0):
        x, y = cdp_x, cdp_y
    else:
        x = np.asarray(f.attributes(segyio.TraceField.SourceX)[:], dtype=np.float64)
        y = np.asarray(f.attributes(segyio.TraceField.SourceY)[:], dtype=np.float64)

    scalar = _coordinate_scalar(int(f.header[0][segyio.TraceField.SourceGroupScalar]))
    return x * scalar, y * scalar


def extract_2d_geometry(f: segyio.SegyFile) -> SectionGeometry:
    """Geometry of a 2D line opened with ignore_geometry=True."""
    n_traces = f.tracecount
    n_samples = len(f.samples)
    dt_ms = _read_dt_ms(f)
    t0_ms = float(f.header[0][segyio.TraceField.DelayRecordingTime])

    cdp = np.asarray(f.attributes(segyio.TraceField.CDP)[:], dtype=np.int64)
    world_x, world_y = _read_world_coords(f)

    if np.any(world_x != 0) or np.any(world_y != 0):
        steps = np.hypot(np.diff(world_x), np.diff(world_y))
        distance = np.concatenate([[0.0], np.cumsum(steps)])
        valid = steps[steps > 0]
        spacing = float(np.median(valid)) if valid.size else 25.0
    else:
        # No navigation in the headers: fall back to unit-index spacing.
        spacing = 25.0
        distance = np.arange(n_traces, dtype=np.float64) * spacing

    return SectionGeometry(
        kind="2d",
        n_traces=n_traces,
        n_samples=n_samples,
        dt_ms=dt_ms,
        t0_ms=t0_ms,
        cdp=cdp,
        world_x=world_x,
        world_y=world_y,
        distance_m=distance,
        trace_spacing_m=spacing,
    )


def extract_3d_geometry(f: segyio.SegyFile) -> SectionGeometry:
    """Geometry of a 3D volume opened with segyio structured mode."""
    ilines = np.asarray(f.ilines, dtype=np.int64)
    xlines = np.asarray(f.xlines, dtype=np.int64)
    dt_ms = _read_dt_ms(f)
    t0_ms = float(f.header[0][segyio.TraceField.DelayRecordingTime])

    world_x, world_y = _read_world_coords(f)
    il_spacing, xl_spacing = _estimate_bin_spacing(
        world_x, world_y, len(ilines), len(xlines)
    )

    return SectionGeometry(
        kind="3d",
        n_traces=f.tracecount,
        n_samples=len(f.samples),
        dt_ms=dt_ms,
        t0_ms=t0_ms,
        ilines=ilines,
        xlines=xlines,
        iline_spacing_m=il_spacing,
        xline_spacing_m=xl_spacing,
    )


def _estimate_bin_spacing(
    world_x: np.ndarray,
    world_y: np.ndarray,
    n_ilines: int,
    n_xlines: int,
) -> tuple[float, float]:
    """Median inline/crossline bin sizes from trace world coordinates.

    Assumes inline-sorted traces (segyio's structured order): consecutive
    traces step along the crossline axis; traces n_xlines apart step
    along the inline axis.
    """
    if not (np.any(world_x != 0) or np.any(world_y != 0)):
        return 25.0, 25.0

    def _median_step(stride: int) -> float:
        # Distance between traces `stride` apart in file order; the median
        # is robust to the wrap-around pairs at inline boundaries.
        dx = world_x[stride:] - world_x[:-stride]
        dy = world_y[stride:] - world_y[:-stride]
        steps = np.hypot(dx, dy)
        valid = steps[steps > 0]
        return float(np.median(valid)) if valid.size else 25.0

    xl_spacing = _median_step(1) if n_xlines > 1 else 25.0
    il_spacing = _median_step(n_xlines) if n_ilines > 1 else 25.0
    return il_spacing, xl_spacing
