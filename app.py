"""Streamlit viewer for geometry-aware SAM 3 seismic segmentation.

Run with:
    streamlit run app.py

Two modes, side by side as tabs:

- **Automatic (text prompts)** - the five fixed noun-phrase prompts are
  applied to every tile of the section (no interaction needed).
- **Interactive (point picking)** - click positive/negative points
  directly on the section; SAM 3's tracker head segments the object you
  indicated. Picks are reported in physical coordinates (CDP / inline /
  crossline and time in ms) read from the SEG-Y headers.

Sections are displayed in true seismic orientation (time down) and all
exports carry the geometry extracted from the file, so ParaView overlays
line up with the seismic.
"""

from __future__ import annotations

import io
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from seismic_app import config
from seismic_app.geometry import SectionGeometry
from seismic_app.inference import (
    Sam3PointSegmenter,
    Sam3SeismicSegmenter,
    Sam3VolumePropagator,
)
from seismic_app.logutil import get_logger
from seismic_app.pipeline import run_on_section, run_on_volume
from seismic_app.preprocessing import inline_to_rgb_25d, normalize_to_uint8, to_rgb
from seismic_app.sgy_loader import load_any
from seismic_app.visualization import overlay_masks
from seismic_app.vtk_export import export_masks

log = get_logger("app")

DATA_DIR = Path("data")

POSITIVE_COLOR = (0, 230, 0)
NEGATIVE_COLOR = (255, 40, 40)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading SAM 3 text-prompt model (first run only)...")
def load_segmenter(checkpoint: str, device: str | None) -> Sam3SeismicSegmenter:
    log.info("App requested text-prompt model (checkpoint=%s, device=%s)", checkpoint, device)
    return Sam3SeismicSegmenter(checkpoint=checkpoint, device=device)


@st.cache_resource(show_spinner="Loading SAM 3 tracker (point prompts, first run only)...")
def load_point_segmenter(checkpoint: str, device: str | None) -> Sam3PointSegmenter:
    log.info("App requested point-tracker model (checkpoint=%s, device=%s)", checkpoint, device)
    return Sam3PointSegmenter(checkpoint=checkpoint, device=device)


@st.cache_resource(show_spinner="Loading SAM 3 video tracker (volume propagation)...")
def load_propagator(checkpoint: str, device: str | None) -> Sam3VolumePropagator:
    log.info("App requested volume propagator (checkpoint=%s, device=%s)", checkpoint, device)
    return Sam3VolumePropagator(checkpoint=checkpoint, device=device)


@st.cache_resource(show_spinner="Normalizing amplitudes...")
def load_normalized(path: str) -> np.ndarray:
    """Percentile-normalized uint8 copy of the file, cached per path."""
    data, _ = load_file(path)
    log.info("Normalizing amplitudes for display (%s values)...", data.size)
    return normalize_to_uint8(data)


@st.cache_resource(show_spinner="Reading SEG-Y file...")
def load_file(path: str) -> tuple[np.ndarray, SectionGeometry]:
    log.info("Loading SEG-Y file: %s", path)
    data, geometry = load_any(path)
    if geometry.kind == "2d":
        log.info(
            "Loaded 2D line shape=%s, dt=%.3g ms, spacing=%.1f m, CDP %s-%s",
            data.shape,
            geometry.dt_ms,
            geometry.trace_spacing_m,
            geometry.cdp[0],
            geometry.cdp[-1],
        )
    else:
        log.info(
            "Loaded 3D volume shape=%s, dt=%.3g ms, %d inlines x %d crosslines",
            data.shape,
            geometry.dt_ms,
            len(geometry.ilines),
            len(geometry.xlines),
        )
    return data, geometry


def list_sgy_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".sgy")


# --------------------------------------------------------------------------
# Slice handling (2D files have exactly one "slice"; 3D volumes have many)
# --------------------------------------------------------------------------


def current_slice_rgb(
    data_u8: np.ndarray,
    geometry: SectionGeometry,
    slice_axis: str,
    slice_idx: int,
) -> np.ndarray:
    """(H, W, 3) uint8 display image of the active slice, time down.

    data_u8 is the pre-normalized uint8 array from load_normalized().
    """
    if geometry.kind == "2d":
        return to_rgb(data_u8)

    if slice_axis == "inline":
        return inline_to_rgb_25d(data_u8, slice_idx)
    if slice_axis == "crossline":
        return to_rgb(data_u8[:, slice_idx, :].T)  # (n_samples, n_ilines)
    return to_rgb(data_u8[:, :, slice_idx])  # time slice: (n_ilines, n_xlines)


def slice_volume_mask(
    vol_mask: np.ndarray,  # (n_il, n_samples, n_xl)
    slice_axis: str,
    slice_idx: int,
) -> np.ndarray:
    """Cut a saved 3D mask down to the currently displayed slice."""
    if slice_axis == "inline":
        return vol_mask[slice_idx]  # (n_samples, n_xl)
    if slice_axis == "crossline":
        return vol_mask[:, :, slice_idx].T  # (n_samples, n_il)
    return vol_mask[:, slice_idx, :]  # time slice: (n_il, n_xl)


def describe_pick(
    geometry: SectionGeometry,
    slice_axis: str,
    slice_idx: int,
    col: int,
    row: int,
) -> str:
    """Physical location of a clicked (col, row) pixel on the active slice."""
    if geometry.kind == "2d":
        return geometry.describe_point(col, row)
    if slice_axis == "inline":
        il = geometry.ilines[slice_idx]
        xl = geometry.xlines[min(col, len(geometry.xlines) - 1)]
        return f"IL {il} / XL {xl} @ {geometry.sample_to_time_ms(row):.0f} ms"
    if slice_axis == "crossline":
        xl = geometry.xlines[slice_idx]
        il = geometry.ilines[min(col, len(geometry.ilines) - 1)]
        return f"IL {il} / XL {xl} @ {geometry.sample_to_time_ms(row):.0f} ms"
    il = geometry.ilines[min(row, len(geometry.ilines) - 1)]
    xl = geometry.xlines[min(col, len(geometry.xlines) - 1)]
    return f"IL {il} / XL {xl} @ {geometry.sample_to_time_ms(slice_idx):.0f} ms"


def place_slice_mask_in_volume(
    mask_2d: np.ndarray,
    volume_shape: tuple[int, int, int],  # (n_il, n_samples, n_xl)
    slice_axis: str,
    slice_idx: int,
) -> np.ndarray:
    """Embed a per-slice mask into a full-volume boolean mask."""
    vol = np.zeros(volume_shape, dtype=bool)
    if slice_axis == "inline":
        vol[slice_idx] = mask_2d  # (n_samples, n_xl)
    elif slice_axis == "crossline":
        vol[:, :, slice_idx] = mask_2d.T  # (n_samples, n_il) -> (n_il, n_samples)
    else:  # time slice: mask is (n_il, n_xl)
        vol[:, slice_idx, :] = mask_2d
    return vol


def store_interactive_mask(
    mask: np.ndarray,
    name: str,
    data: np.ndarray,
    geometry: SectionGeometry,
    slice_axis: str,
    slice_idx: int,
) -> str:
    """Store a point-picked mask without allocating a full volume per click."""
    name = name.strip() or "picked object"
    if geometry.kind == "3d":
        vol_shape = (data.shape[0], data.shape[2], data.shape[1])
        st.session_state.interactive_masks[name] = {
            "kind": "slice",
            "mask": mask,
            "axis": slice_axis,
            "index": int(slice_idx),
            "volume_shape": vol_shape,
        }
    else:
        st.session_state.interactive_masks[name] = mask
    coverage = 100.0 * float(mask.mean())
    log.info("Saved interactive mask '%s' for export (coverage=%.2f%%)", name, coverage)
    return name


def materialize_interactive_mask(value: object) -> np.ndarray:
    """Convert a lightweight saved slice into its exportable volume mask."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict) and value.get("kind") == "slice":
        return place_slice_mask_in_volume(
            value["mask"],
            tuple(value["volume_shape"]),
            value["axis"],
            int(value["index"]),
        )
    if isinstance(value, dict) and value.get("kind") == "frames":
        frames = value["masks"]
        if value["axis"] == "inline":
            return frames
        if value["axis"] == "crossline":
            return np.transpose(frames, (2, 1, 0))
        return np.transpose(frames, (1, 0, 2))
    raise TypeError(f"Unsupported interactive mask value: {type(value)!r}")


def saved_mask_for_slice(
    value: object,
    slice_axis: str,
    slice_idx: int,
) -> np.ndarray | None:
    """Return only the visible 2D cut of a saved mask."""
    if isinstance(value, dict) and value.get("kind") == "slice":
        if value["axis"] == slice_axis and int(value["index"]) == slice_idx:
            return value["mask"]
        return None
    if isinstance(value, dict) and value.get("kind") == "frames":
        volume_view = materialize_interactive_mask(value)
        return slice_volume_mask(volume_view, slice_axis, slice_idx)
    if isinstance(value, np.ndarray):
        if value.ndim == 3:
            return slice_volume_mask(value, slice_axis, slice_idx)
        return value
    return None


# --------------------------------------------------------------------------
# Point-picking helpers
# --------------------------------------------------------------------------


def draw_point_markers(
    image: Image.Image,
    points: list[dict],
    scale_x: float,
    scale_y: float,
) -> Image.Image:
    """Draw positive (green) / negative (red) pick markers on a copy."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    r = 6
    for pt in points:
        cx, cy = pt["col"] * scale_x, pt["row"] * scale_y
        color = POSITIVE_COLOR if pt["label"] == 1 else NEGATIVE_COLOR
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
        draw.line([cx - r - 3, cy, cx + r + 3, cy], fill=color, width=1)
        draw.line([cx, cy - r - 3, cx, cy + r + 3], fill=color, width=1)
    return annotated


def slice_state_key(slice_axis: str, slice_idx: int) -> str:
    return f"{slice_axis}:{slice_idx}"


def display_to_slice_coordinates(
    x: float,
    y: float,
    display_size: tuple[int, int],
    slice_size: tuple[int, int],
) -> tuple[int, int]:
    """Map displayed image pixels to clamped full-resolution (col, row)."""
    display_width, display_height = display_size
    slice_width, slice_height = slice_size
    scale_x = display_width / slice_width
    scale_y = display_height / slice_height
    col = int(np.clip(round(x / scale_x), 0, slice_width - 1))
    row = int(np.clip(round(y / scale_y), 0, slice_height - 1))
    return col, row


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="SAM 3 Seismic Segmentation", layout="wide")
    st.title("SAM 3 Seismic Segmentation")
    st.caption(
        "Geometry-aware segmentation of .sgy seismic data: automatic "
        "text-prompt detection of faults, channels, facies, salt bodies and "
        "horizons, plus interactive point-picked segmentation."
    )

    # ---- sidebar: file, model, geometry -----------------------------------
    with st.sidebar:
        st.header("Settings")
        sgy_files = list_sgy_files()
        if not sgy_files:
            st.warning(f"No .sgy files found in {DATA_DIR}/. Add some and reload.")
            return
        options = [str(p) for p in sgy_files]
        # Default to the 3D volume so slice navigation/propagation show up.
        default_idx = next(
            (i for i, p in enumerate(options) if "SEGY0000" in p), 0
        )
        selected = st.selectbox("Seismic file", options, index=default_idx)
        checkpoint = st.text_input("SAM 3 checkpoint", value=config.DEFAULT_CHECKPOINT)
        device_options = ["auto", "cpu"]
        if torch.cuda.is_available():
            device_options.insert(1, "cuda")
        device = st.selectbox("Device", device_options, index=0)
        if not torch.cuda.is_available():
            st.caption("CUDA is unavailable in the active Python environment; using CPU.")
        device_arg = None if device == "auto" else device
        threshold = st.slider(
            "Mask probability threshold", 0.0, 1.0, config.MASK_THRESHOLD, 0.05
        )
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)

    # ---- session defaults ---------------------------------------------------
    for key, default in (
        ("text_result", None),
        ("points", None),
        ("last_click", None),
        ("point_mask", None),
        ("interactive_masks", None),
        ("propagation_review", None),
        ("prop_frame", None),
        ("display_base_cache", None),
        ("point_preview_started", None),
        ("point_preview_latency", None),
        ("export_payloads", None),
    ):
        if key not in st.session_state:
            if key == "points":
                st.session_state[key] = []
            elif key == "interactive_masks":
                st.session_state[key] = {}
            elif key == "display_base_cache":
                st.session_state[key] = {}
            else:
                st.session_state[key] = default

    # ---- load file, reset per-file state -----------------------------------
    data, geometry = load_file(selected)
    if st.session_state.get("loaded_path") != selected:
        log.info("Active file changed to %s", selected)
        st.session_state.loaded_path = selected
        st.session_state.text_result = None
        st.session_state.points = []
        st.session_state.last_click = None
        st.session_state.point_mask = None
        st.session_state.interactive_masks = {}
        st.session_state.propagation_review = None
        st.session_state.prop_frame = None
        st.session_state.display_base_cache = {}
        st.session_state.export_payloads = None

    # ---- sidebar: geometry read from the headers ---------------------------
    with st.sidebar:
        st.divider()
        st.subheader("Geometry (from SEG-Y headers)")
        if geometry.kind == "2d":
            st.markdown(
                f"- **Type:** 2D line, {geometry.n_traces} traces x "
                f"{geometry.n_samples} samples\n"
                f"- **Sample interval:** {geometry.dt_ms:g} ms "
                f"(delay {geometry.t0_ms:g} ms)\n"
                f"- **Trace spacing:** {geometry.trace_spacing_m:.1f} m (median, "
                "from trace coordinates)\n"
                f"- **CDP range:** {geometry.cdp[0]} - {geometry.cdp[-1]}\n"
                f"- **Line length:** {geometry.distance_m[-1]:,.0f} m"
            )
        else:
            st.markdown(
                f"- **Type:** 3D volume, {len(geometry.ilines)} inlines x "
                f"{len(geometry.xlines)} crosslines x {geometry.n_samples} samples\n"
                f"- **Sample interval:** {geometry.dt_ms:g} ms "
                f"(delay {geometry.t0_ms:g} ms)\n"
                f"- **Bin size:** {geometry.iline_spacing_m:.1f} m (IL) x "
                f"{geometry.xline_spacing_m:.1f} m (XL)\n"
                f"- **Inlines:** {geometry.ilines[0]} - {geometry.ilines[-1]}, "
                f"**Crosslines:** {geometry.xlines[0]} - {geometry.xlines[-1]}"
            )
        with st.expander("Override for export"):
            spacing_override = st.number_input(
                "Trace spacing (m)", min_value=0.0, value=0.0, step=1.0,
                help="0 = use the value measured from the headers.",
            )
            dt_override = st.number_input(
                "Sample interval (ms)", min_value=0.0, value=0.0, step=0.5,
                help="0 = use the value read from the headers.",
            )
            if spacing_override > 0:
                geometry.trace_spacing_m = spacing_override
            if dt_override > 0:
                geometry.dt_ms = dt_override

    # ---- 3D slice navigation ------------------------------------------------
    slice_axis, slice_idx = "line", 0
    if geometry.kind == "3d":
        with st.sidebar:
            st.divider()
            st.subheader("Slice navigation")
            slice_axis = st.selectbox("Slice axis", ["inline", "crossline", "time slice"])
            if slice_axis == "inline":
                n = len(geometry.ilines)
                i = st.slider("Inline", 0, n - 1, n // 2)
                st.caption(f"Inline number {geometry.ilines[i]}")
            elif slice_axis == "crossline":
                n = len(geometry.xlines)
                i = st.slider("Crossline", 0, n - 1, n // 2)
                st.caption(f"Crossline number {geometry.xlines[i]}")
            else:
                n = geometry.n_samples
                i = st.slider("Time sample", 0, n - 1, n // 2)
                st.caption(f"t = {geometry.sample_to_time_ms(i):.0f} ms")
            slice_idx = i

    data_u8 = load_normalized(selected)
    slice_rgb = current_slice_rgb(data_u8, geometry, slice_axis, slice_idx)
    slice_h, slice_w = slice_rgb.shape[:2]

    tab_auto, tab_pick = st.tabs(
        ["Automatic (text prompts)", "Interactive (point picking)"]
    )

    # ======================================================================
    # Tab 1: automatic text-prompt segmentation
    # ======================================================================
    with tab_auto:
        run_clicked = st.button("Run segmentation", type="primary")
        if run_clicked:
            log.info("=== Automatic text-prompt segmentation requested for %s ===", selected)
            try:
                segmenter = load_segmenter(checkpoint, device_arg)
                with st.spinner(f"Segmenting {selected} ... (see terminal for progress)"):
                    if geometry.kind == "3d":
                        result = run_on_volume(
                            data, geometry, segmenter, threshold=threshold
                        )
                    else:
                        result = run_on_section(
                            data, geometry, segmenter, threshold=threshold
                        )
                st.session_state.text_result = result
                st.session_state.export_payloads = None
                log.info("Automatic segmentation complete for %s", selected)
            except Exception as exc:
                log.error("Automatic segmentation FAILED: %s", exc)
                log.error(traceback.format_exc())
                st.error(f"Segmentation failed: {exc}")
                st.caption("Check the terminal for the full traceback.")

        result = st.session_state.text_result
        if result is None:
            st.info("Click **Run segmentation** to apply the five fixed text prompts.")
        else:
            layer_cols = st.columns(len(config.LABEL_STYLES))
            layer_toggles = {}
            for col, style in zip(layer_cols, config.LABEL_STYLES):
                with col:
                    layer_toggles[style.noun_phrase] = st.checkbox(
                        style.noun_phrase, value=True, key=f"layer_{style.noun_phrase}"
                    )
            active_layers = [name for name, on in layer_toggles.items() if on]

            if geometry.kind == "3d":
                # Show the masks on the active inline (text mode segments
                # inline sections, so other axes show the same volume).
                if slice_axis == "inline":
                    shown_masks = {p: m[slice_idx] for p, m in result.masks.items()}
                    shown_rgb = slice_rgb
                else:
                    mid = data.shape[0] // 2
                    shown_masks = {p: m[mid] for p, m in result.masks.items()}
                    shown_rgb = result.rgb
                    st.caption("Overlay shown on the middle inline; switch the "
                               "slice axis to 'inline' to browse.")
            else:
                shown_masks = result.masks
                shown_rgb = result.rgb

            overlay = overlay_masks(shown_rgb, shown_masks, alpha=alpha, only=active_layers)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original section")
                st.image(shown_rgb, width="stretch")
            with col2:
                st.subheader("Segmentation overlay")
                st.image(np.array(overlay.convert("RGB")), width="stretch")

            st.subheader("Detected coverage per feature")
            cols = st.columns(len(config.LABEL_STYLES))
            for col, style in zip(cols, config.LABEL_STYLES):
                mask = result.masks.get(style.noun_phrase)
                pct = 100.0 * mask.mean() if mask is not None else 0.0
                col.metric(style.noun_phrase, f"{pct:.2f}%")

    # ======================================================================
    # Tab 2: interactive point picking
    # ======================================================================
    with tab_pick:
        # ---- Propagation review (SAM2-style frame scrubber) -----------------
        # After "Propagate through volume", the per-slice masks are kept as a
        # frame stack so you can scrub forward/backward and see the tracking
        # result - the sidebar alone was easy to miss and went through a
        # volume reshape that did not feel like video playback.
        review = st.session_state.get("propagation_review")
        display_axis = slice_axis
        display_idx = slice_idx
        if review is not None:
            st.subheader("Review propagation")
            st.caption(
                f"Scrub along **{review['axis']}** to see how the mask "
                f"tracked across the volume (SAM 2 video-style). "
                f"Anchor was frame {review['anchor']}."
            )
            n_rev = int(review["masks"].shape[0])
            default_frame = st.session_state.get("prop_frame")
            if default_frame is None or not (0 <= default_frame < n_rev):
                default_frame = int(review["anchor"])
            display_idx = st.slider(
                f"Propagated frame ({review['axis']})",
                0,
                n_rev - 1,
                value=default_frame,
                key="prop_review_scrub",
            )
            st.session_state.prop_frame = display_idx
            display_axis = review["axis"]
            frame_cov = 100.0 * float(review["masks"][display_idx].mean())
            st.caption(
                f"Frame {display_idx} / {n_rev - 1} — coverage on this slice "
                f"{frame_cov:.2f}% (total volume "
                f"{100.0 * float(review['masks'].mean()):.2f}%)"
            )
            if st.button("Clear propagation review"):
                st.session_state.propagation_review = None
                st.session_state.prop_frame = None
                st.rerun()

        # Recompute the displayed slice when the review scrubber overrides
        # the sidebar navigation (same file, different frame index/axis).
        if display_axis != slice_axis or display_idx != slice_idx:
            slice_rgb = current_slice_rgb(data_u8, geometry, display_axis, display_idx)
            slice_h, slice_w = slice_rgb.shape[:2]
            slice_axis, slice_idx = display_axis, display_idx

        if geometry.kind == "2d":
            st.caption(
                "This file is a single 2D line: one cross-section whose "
                "vertical axis is the **entire time range** "
                f"({geometry.n_samples} samples x {geometry.dt_ms:g} ms = "
                f"{geometry.sample_to_time_ms(geometry.n_samples - 1):,.0f} ms). "
                "It is the seismic equivalent of one video frame, so there is "
                "nothing to propagate through - the mask you make here already "
                "covers all of time and all traces on this line. Propagation "
                "through inlines (like SAM 2 video) applies to 3D volumes."
            )

        ctl1, ctl2, ctl3, ctl4 = st.columns([2, 1, 1, 2])
        with ctl1:
            click_label = st.radio(
                "Click adds", ["positive point", "negative point"],
                horizontal=True, label_visibility="collapsed",
            )
            live_preview = st.checkbox(
                "Live mask preview (segment on every click)", value=True,
                help="Like the SAM 2 demo: the mask updates as soon as you "
                "click. The image is encoded once and cached, so the first "
                "click is slow (full encoder) and later clicks are fast.",
            )
            mask_name = st.text_input(
                "Mask label (used in Export / ParaView)",
                value="picked object",
                key="mask_label_name",
            )
        with ctl2:
            if st.button("Undo point") and st.session_state.points:
                removed = st.session_state.points.pop()
                log.info("Undid point at col=%s row=%s", removed["col"], removed["row"])
                st.session_state.point_mask = None
        with ctl3:
            if st.button("Clear points"):
                log.info("Cleared %d points", len(st.session_state.points))
                st.session_state.points = []
                st.session_state.point_mask = None
        with ctl4:
            display_width = st.slider(
                "Display width (px)", 300, 1600,
                int(np.clip(slice_w * 3, 400, 1000)), 50,
            )
            display_height = st.slider(
                "Display height (px)", 300, 3000,
                int(np.clip(slice_h, 400, 900)), 50,
                help="Seismic sections are usually much taller than wide; "
                "compress the time axis here for comfortable picking. Click "
                "coordinates are mapped back to true samples either way.",
            )

        # Only picks on the current slice are shown/used. The key includes
        # the file stem so switching files never reuses another file's
        # click-component state (which used to replay a stale click onto
        # the newly selected file).
        skey = f"{Path(selected).stem}:{slice_state_key(slice_axis, slice_idx)}"
        image_key = (str(Path(selected).resolve()), slice_axis, int(slice_idx))

        point_tracker: Sam3PointSegmenter | None = None
        if live_preview:
            try:
                point_tracker = load_point_segmenter(checkpoint, device_arg)
                with st.spinner("Preparing active slice for instant point prompts..."):
                    point_tracker.prepare_image(slice_rgb, image_key=image_key)
                st.caption("Interactive model ready; point clicks use cached image features.")
            except Exception as exc:
                log.error("Interactive model preparation failed: %s", exc)
                st.warning(f"Live preview is not ready: {exc}")

        def _segment_current_points() -> bool:
            """Segment from all points on this slice; save mask for export."""
            pts = [p for p in st.session_state.points if p["slice"] == skey]
            if not pts:
                return False
            log.info(
                "=== Point segmentation (%d point(s) on %s) ===", len(pts), selected
            )
            try:
                tracker = point_tracker or load_point_segmenter(checkpoint, device_arg)
                with st.spinner(
                    "Segmenting from points... (see terminal for progress)"
                ):
                    mask = tracker.segment(
                        slice_rgb,
                        [(p["col"], p["row"]) for p in pts],
                        [p["label"] for p in pts],
                        image_key=image_key,
                    )
                st.session_state.point_inference_timings = dict(tracker.last_timings)
            except Exception as exc:
                log.error("Point segmentation FAILED: %s", exc)
                log.error(traceback.format_exc())
                st.error(f"Point segmentation failed: {exc}")
                st.caption("Check the terminal for the full traceback.")
                return False
            st.session_state.point_mask = mask
            st.session_state.point_mask_slice = skey
            store_interactive_mask(
                mask, mask_name, data, geometry, slice_axis, slice_idx
            )
            st.session_state.export_payloads = None
            return True

        slice_points = [p for p in st.session_state.points if p["slice"] == skey]

        # --- render the clickable section at a known display scale ---------
        # Overlay every saved interactive mask (sliced to the current view,
        # so a propagated volume mask stays visible while you browse
        # inlines/crosslines/time slices), plus the live pick preview.
        # When a propagation review stack is active, prefer its per-frame
        # mask for that frame - that is the SAM2-style scrubbing view.
        scale_x = display_width / slice_w
        scale_y = display_height / slice_h
        overlay_dict: dict[str, np.ndarray] = {}
        review = st.session_state.get("propagation_review")
        if (
            review is not None
            and review["axis"] == slice_axis
            and 0 <= slice_idx < review["masks"].shape[0]
        ):
            overlay_dict[review["name"]] = review["masks"][slice_idx]
        for name, m in st.session_state.interactive_masks.items():
            if name in overlay_dict:
                continue  # already showing the review-frame version
            shown = saved_mask_for_slice(m, slice_axis, slice_idx)
            if shown is not None:
                overlay_dict[name] = shown
        if (
            st.session_state.point_mask is not None
            and st.session_state.get("point_mask_slice") == skey
        ):
            overlay_dict["current pick"] = st.session_state.point_mask
        display_cache_key = (
            str(Path(selected).resolve()),
            slice_axis,
            int(slice_idx),
            int(display_width),
            int(display_height),
        )
        display_cache = st.session_state.display_base_cache
        if display_cache_key not in display_cache:
            display_cache[display_cache_key] = Image.fromarray(slice_rgb).resize(
                (display_width, display_height), Image.Resampling.BILINEAR
            )
            while len(display_cache) > 8:
                display_cache.pop(next(iter(display_cache)))
        base_img = display_cache[display_cache_key]
        if overlay_dict:
            display_masks = {
                name: np.asarray(
                    Image.fromarray(mask).resize(
                        (display_width, display_height), Image.Resampling.NEAREST
                    ),
                    dtype=bool,
                )
                for name, mask in overlay_dict.items()
            }
            disp_img = overlay_masks(
                np.asarray(base_img), display_masks, alpha=alpha
            ).convert("RGB")
        else:
            disp_img = base_img.copy()
        disp_img = draw_point_markers(disp_img, slice_points, scale_x, scale_y)
        if (
            st.session_state.point_preview_started is not None
            and st.session_state.point_mask is not None
        ):
            latency = time.perf_counter() - st.session_state.point_preview_started
            st.session_state.point_preview_latency = latency
            st.session_state.point_preview_started = None
            log.info("Click-to-rendered-preview path completed in %.3fs", latency)

        st.caption(
            "Click on the section to add points. Time runs down, traces "
            "across - exactly as the data is organised in the file."
        )
        click = streamlit_image_coordinates(disp_img, key=f"picker_{skey}")

        if click is not None and click != st.session_state.last_click:
            st.session_state.last_click = click
            col, row = display_to_slice_coordinates(
                click["x"],
                click["y"],
                (display_width, display_height),
                (slice_w, slice_h),
            )
            label = 1 if click_label == "positive point" else 0
            st.session_state.points.append(
                {
                    "col": col,
                    "row": row,
                    "label": label,
                    "slice": skey,
                }
            )
            log.info(
                "Point pick: %s at %s (display %d,%d -> col=%d row=%d)",
                "positive" if label == 1 else "negative",
                describe_pick(geometry, slice_axis, slice_idx, col, row),
                click["x"],
                click["y"],
                col,
                row,
            )
            st.session_state.point_mask = None
            if live_preview:
                # SAM2-demo behavior: refresh the mask on every click.
                st.session_state.point_preview_started = time.perf_counter()
                _segment_current_points()
            st.rerun()

        if st.session_state.point_preview_latency is not None:
            timings = st.session_state.get("point_inference_timings", {})
            st.caption(
                f"Last preview: {st.session_state.point_preview_latency:.2f}s end-to-end "
                f"(decoder {timings.get('prompt_decode', 0.0):.2f}s, "
                f"post-process {timings.get('post_process', 0.0):.2f}s)"
            )

        if slice_points:
            st.markdown("**Picked points (physical coordinates):**")
            for i, pt in enumerate(slice_points):
                sign = "+" if pt["label"] == 1 else "-"
                st.markdown(
                    f"{i + 1}. `{sign}` "
                    f"{describe_pick(geometry, slice_axis, slice_idx, pt['col'], pt['row'])}"
                )

            seg1, seg2 = st.columns([1, 1])
            with seg1:
                segment_clicked = st.button(
                    "Segment from points", type="primary",
                    help="Re-runs the mask from all current points. With live "
                    "preview on, this happens automatically per click.",
                )
            with seg2:
                propagate_clicked = False
                if geometry.kind == "3d":
                    propagate_clicked = st.button(
                        f"Propagate through volume ({slice_axis} direction)",
                        help="SAM 2 video-style: treats the slices along this "
                        "axis as consecutive frames and tracks the picked "
                        "object through the whole cube, both directions.",
                    )

            if segment_clicked:
                if _segment_current_points():
                    st.success(
                        f"Mask '{mask_name}' updated and saved. Keep clicking to "
                        "refine, or scroll down to **Export** for PNG / NPZ / "
                        "ParaView (.vts / .vti) downloads."
                    )
                    st.rerun()

            if propagate_clicked:
                log.info(
                    "=== Volume propagation requested along %s from slice %d ===",
                    slice_axis,
                    slice_idx,
                )
                try:
                    propagator = load_propagator(checkpoint, device_arg)
                    if slice_axis == "inline":
                        n_frames = data.shape[0]
                    elif slice_axis == "crossline":
                        n_frames = data.shape[1]
                    else:
                        n_frames = data.shape[2]
                    with st.spinner(
                        f"Propagating through {n_frames} slices... "
                        "(one tracker pass per slice - watch the terminal)"
                    ):
                        progress_bar = st.progress(0.0, text="Preparing frame stack...")
                        progress_slot = st.empty()
                        prep_started = time.perf_counter()
                        frames = [
                            current_slice_rgb(data_u8, geometry, slice_axis, i)
                            for i in range(n_frames)
                        ]
                        prep_elapsed = time.perf_counter() - prep_started

                        def _update_progress(
                            done: int,
                            total: int,
                            frame_idx: int,
                            frame_mask: np.ndarray,
                        ) -> None:
                            progress_bar.progress(
                                done / total,
                                text=f"Tracked {done}/{total} slices",
                            )
                            if done == 1 or done == total or done % max(1, total // 20) == 0:
                                progress_slot.caption(
                                    f"Latest frame {frame_idx}: "
                                    f"{100.0 * float(frame_mask.mean()):.2f}% coverage"
                                )

                        frame_masks = propagator.propagate(
                            frames,
                            anchor_idx=slice_idx,
                            points=[(p["col"], p["row"]) for p in slice_points],
                            labels=[p["label"] for p in slice_points],
                            progress=_update_progress,
                        )
                        progress_bar.progress(1.0, text="Propagation complete")
                        st.session_state.propagation_timings = {
                            "frame_prep": prep_elapsed,
                            **propagator.last_timings,
                        }
                    vol_shape = (data.shape[0], data.shape[2], data.shape[1])
                    base_name = mask_name.strip() or "picked object"
                    name = f"{base_name} (volume)"
                    # Drop the single-slice preview of the same label so
                    # scrubbing is not confused with a one-frame-only mask.
                    st.session_state.interactive_masks.pop(base_name, None)
                    st.session_state.interactive_masks[name] = {
                        "kind": "frames",
                        "masks": frame_masks,
                        "axis": slice_axis,
                        "volume_shape": vol_shape,
                    }
                    st.session_state.export_payloads = None
                    # Keep the raw per-frame stack for SAM2-style scrubbing
                    # in the Interactive tab (does not depend on volume reshape).
                    n_live = int(np.count_nonzero(frame_masks.reshape(n_frames, -1).any(axis=1)))
                    st.session_state.propagation_review = {
                        "axis": slice_axis,
                        "masks": frame_masks,
                        "name": name,
                        "anchor": int(slice_idx),
                    }
                    st.session_state.prop_frame = int(slice_idx)
                    st.session_state.point_mask = None
                    st.session_state.points = [
                        p for p in st.session_state.points if p["slice"] != skey
                    ]
                    log.info(
                        "Volume propagation saved as '%s' "
                        "(%d/%d frames have a non-empty mask)",
                        name,
                        n_live,
                        n_frames,
                    )
                    st.success(
                        f"Propagated through {n_frames} slices ({n_live} with "
                        f"mask). Use the **Review propagation** slider above "
                        "to scrub forward/backward like a video."
                    )
                    st.rerun()
                except Exception as exc:
                    log.error("Volume propagation FAILED: %s", exc)
                    log.error(traceback.format_exc())
                    st.error(f"Volume propagation failed: {exc}")
                    st.caption("Check the terminal for the full traceback.")
        else:
            if review is None:
                st.info(
                    "No points on this slice yet - click on the image above. "
                    "With live preview on, the mask appears right after your "
                    "first click. After you Propagate, a review slider appears "
                    "so you can scrub through every frame."
                )

        if st.session_state.interactive_masks:
            st.markdown(
                "**Saved interactive masks (available in Export):** "
                + ", ".join(f"`{k}`" for k in st.session_state.interactive_masks)
            )
            if st.button("Discard all interactive masks"):
                log.info(
                    "Discarded interactive masks: %s",
                    list(st.session_state.interactive_masks),
                )
                st.session_state.interactive_masks = {}
                st.session_state.point_mask = None
                st.session_state.propagation_review = None
                st.session_state.prop_frame = None
                st.session_state.export_payloads = None
                st.rerun()

    # ======================================================================
    # Export (text-prompt masks + interactive masks, geometry attached)
    # ======================================================================
    st.divider()
    st.subheader("Export")

    has_export_masks = (
        st.session_state.text_result is not None
        or bool(st.session_state.interactive_masks)
    )
    if not has_export_masks:
        st.info(
            "No masks to export yet. Either run **Automatic** segmentation, or in "
            "**Interactive** pick points and click **Segment from points** "
            "(that step also saves the mask for ParaView export)."
        )
        return

    stem = Path(selected).stem
    if st.button(
        "Prepare export files",
        help="Build compressed NPZ and ParaView files only when requested, "
        "so live point clicks stay responsive.",
    ):
        with st.spinner("Materializing masks and preparing export files..."):
            export_masks_dict: dict[str, np.ndarray] = {}
            if st.session_state.text_result is not None:
                export_masks_dict.update(st.session_state.text_result.masks)
            export_masks_dict.update(
                {
                    name: materialize_interactive_mask(value)
                    for name, value in st.session_state.interactive_masks.items()
                }
            )
            prepared: list[tuple[str, bytes, str]] = []
            if geometry.kind == "2d":
                overlay_all = overlay_masks(slice_rgb, export_masks_dict, alpha=alpha)
                png_buffer = io.BytesIO()
                overlay_all.convert("RGB").save(png_buffer, format="PNG")
                prepared.append((".png", png_buffer.getvalue(), "image/png"))

            npz_buffer = io.BytesIO()
            np.savez_compressed(
                npz_buffer,
                **{k.replace(" ", "_"): v for k, v in export_masks_dict.items()},
            )
            prepared.append(
                (".npz", npz_buffer.getvalue(), "application/octet-stream")
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                written = export_masks(
                    export_masks_dict, data, geometry, Path(tmp_dir) / "masks"
                )
                prepared.extend(
                    (p.suffix, p.read_bytes(), "application/octet-stream")
                    for p in written
                )
            st.session_state.export_payloads = prepared
            log.info("Prepared %d export payload(s)", len(prepared))

    for suffix, blob, mime in st.session_state.export_payloads or []:
        label = {
            ".png": "Download overlay PNG",
            ".npz": "Download masks (.npz)",
            ".vts": "Download masks (.vts - world coordinates, for ParaView)",
            ".vti": "Download masks (.vti - regular grid, for ParaView)",
        }.get(suffix, f"Download masks ({suffix})")
        st.download_button(
            label,
            data=blob,
            file_name=(
                f"{stem}_overlay.png" if suffix == ".png" else f"{stem}_masks{suffix}"
            ),
            mime=mime,
            help="Geometry (trace positions, sample interval, delay time) is "
            "read from the SEG-Y headers, so the masks land exactly on the "
            "seismic in ParaView. Threshold by 'label' to isolate features.",
        )


if __name__ == "__main__":
    main()
