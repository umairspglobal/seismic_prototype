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
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from seismic_app import config
from seismic_app.geometry import SectionGeometry
from seismic_app.inference import Sam3PointSegmenter, Sam3SeismicSegmenter
from seismic_app.pipeline import run_on_section, run_on_volume
from seismic_app.preprocessing import inline_to_rgb_25d, normalize_to_uint8, to_rgb
from seismic_app.sgy_loader import load_any
from seismic_app.visualization import overlay_masks
from seismic_app.vtk_export import export_masks

DATA_DIR = Path("data")

POSITIVE_COLOR = (0, 230, 0)
NEGATIVE_COLOR = (255, 40, 40)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading SAM 3 text-prompt model (first run only)...")
def load_segmenter(checkpoint: str, device: str | None) -> Sam3SeismicSegmenter:
    return Sam3SeismicSegmenter(checkpoint=checkpoint, device=device)


@st.cache_resource(show_spinner="Loading SAM 3 tracker (point prompts, first run only)...")
def load_point_segmenter(checkpoint: str, device: str | None) -> Sam3PointSegmenter:
    return Sam3PointSegmenter(checkpoint=checkpoint, device=device)


@st.cache_resource(show_spinner="Reading SEG-Y file...")
def load_file(path: str) -> tuple[np.ndarray, SectionGeometry]:
    return load_any(path)


def list_sgy_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".sgy")


# --------------------------------------------------------------------------
# Slice handling (2D files have exactly one "slice"; 3D volumes have many)
# --------------------------------------------------------------------------


def current_slice_rgb(
    data: np.ndarray,
    geometry: SectionGeometry,
    slice_axis: str,
    slice_idx: int,
) -> np.ndarray:
    """(H, W, 3) uint8 display image of the active slice, time down."""
    if geometry.kind == "2d":
        return to_rgb(normalize_to_uint8(data))

    cube_u8 = normalize_to_uint8(data)
    if slice_axis == "inline":
        return inline_to_rgb_25d(cube_u8, slice_idx)
    if slice_axis == "crossline":
        return to_rgb(cube_u8[:, slice_idx, :].T)  # (n_samples, n_ilines)
    return to_rgb(cube_u8[:, :, slice_idx])  # time slice: (n_ilines, n_xlines)


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
        selected = st.selectbox("Seismic file", [str(p) for p in sgy_files])
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

    # ---- load file, reset per-file state -----------------------------------
    data, geometry = load_file(selected)
    if st.session_state.get("loaded_path") != selected:
        st.session_state.loaded_path = selected
        st.session_state.text_result = None
        st.session_state.points = []
        st.session_state.last_click = None
        st.session_state.point_mask = None
        st.session_state.interactive_masks = {}

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

    slice_rgb = current_slice_rgb(data, geometry, slice_axis, slice_idx)
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
            segmenter = load_segmenter(checkpoint, device_arg)
            with st.spinner(f"Segmenting {selected} ..."):
                if geometry.kind == "3d":
                    result = run_on_volume(data, geometry, segmenter, threshold=threshold)
                else:
                    result = run_on_section(data, geometry, segmenter, threshold=threshold)
            st.session_state.text_result = result

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
        ctl1, ctl2, ctl3, ctl4 = st.columns([2, 1, 1, 2])
        with ctl1:
            click_label = st.radio(
                "Click adds", ["positive point", "negative point"],
                horizontal=True, label_visibility="collapsed",
            )
        with ctl2:
            if st.button("Undo point") and st.session_state.points:
                st.session_state.points.pop()
                st.session_state.point_mask = None
        with ctl3:
            if st.button("Clear points"):
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

        # Only picks on the current slice are shown/used.
        skey = slice_state_key(slice_axis, slice_idx)
        slice_points = [p for p in st.session_state.points if p["slice"] == skey]

        # --- render the clickable section at a known display scale ---------
        scale_x = display_width / slice_w
        scale_y = display_height / slice_h
        base_img = Image.fromarray(slice_rgb)
        if st.session_state.point_mask is not None:
            base_img = overlay_masks(
                slice_rgb, {"picked object": st.session_state.point_mask},
                alpha=alpha,
            ).convert("RGB")
        disp_img = base_img.resize(
            (display_width, display_height), Image.Resampling.BILINEAR
        )
        disp_img = draw_point_markers(disp_img, slice_points, scale_x, scale_y)

        st.caption(
            "Click on the section to add points. Time runs down, traces "
            "across - exactly as the data is organised in the file."
        )
        click = streamlit_image_coordinates(disp_img, key=f"picker_{skey}")

        if click is not None and click != st.session_state.last_click:
            st.session_state.last_click = click
            col = int(np.clip(round(click["x"] / scale_x), 0, slice_w - 1))
            row = int(np.clip(round(click["y"] / scale_y), 0, slice_h - 1))
            st.session_state.points.append(
                {
                    "col": col,
                    "row": row,
                    "label": 1 if click_label == "positive point" else 0,
                    "slice": skey,
                }
            )
            st.session_state.point_mask = None
            st.rerun()

        if slice_points:
            st.markdown("**Picked points (physical coordinates):**")
            for i, pt in enumerate(slice_points):
                sign = "+" if pt["label"] == 1 else "-"
                st.markdown(
                    f"{i + 1}. `{sign}` "
                    f"{describe_pick(geometry, slice_axis, slice_idx, pt['col'], pt['row'])}"
                )

            seg1, seg2, seg3 = st.columns([1, 2, 1])
            with seg1:
                segment_clicked = st.button("Segment from points", type="primary")
            with seg2:
                mask_name = st.text_input(
                    "Label name", value="picked object", label_visibility="collapsed",
                    placeholder="Name for this mask (e.g. 'salt dome')",
                )
            with seg3:
                add_clicked = st.button("Add to layers")

            if segment_clicked:
                tracker = load_point_segmenter(checkpoint, device_arg)
                points_xy = [(p["col"], p["row"]) for p in slice_points]
                labels = [p["label"] for p in slice_points]
                with st.spinner("Segmenting from points..."):
                    mask = tracker.segment(slice_rgb, points_xy, labels)
                st.session_state.point_mask = mask
                st.rerun()

            if add_clicked and st.session_state.point_mask is not None:
                name = mask_name.strip() or "picked object"
                if geometry.kind == "3d":
                    vol_shape = (data.shape[0], data.shape[2], data.shape[1])
                    st.session_state.interactive_masks[name] = place_slice_mask_in_volume(
                        st.session_state.point_mask, vol_shape, slice_axis, slice_idx
                    )
                else:
                    st.session_state.interactive_masks[name] = st.session_state.point_mask
                st.session_state.point_mask = None
                st.session_state.points = [
                    p for p in st.session_state.points if p["slice"] != skey
                ]
                st.success(f"Added mask '{name}' to the export layers.")
                st.rerun()
        else:
            st.info("No points on this slice yet - click on the image above.")

        if st.session_state.interactive_masks:
            st.markdown(
                "**Saved interactive masks:** "
                + ", ".join(f"`{k}`" for k in st.session_state.interactive_masks)
            )
            if st.button("Discard all interactive masks"):
                st.session_state.interactive_masks = {}
                st.rerun()

    # ======================================================================
    # Export (text-prompt masks + interactive masks, geometry attached)
    # ======================================================================
    st.divider()
    st.subheader("Export")

    export_masks_dict: dict[str, np.ndarray] = {}
    if st.session_state.text_result is not None:
        export_masks_dict.update(st.session_state.text_result.masks)
    export_masks_dict.update(st.session_state.interactive_masks)

    if not export_masks_dict:
        st.info("Run the automatic segmentation and/or add interactive masks to export.")
        return

    stem = Path(selected).stem

    # Overlay PNG (2D only - a volume has no single overlay image).
    if geometry.kind == "2d":
        overlay_all = overlay_masks(slice_rgb, export_masks_dict, alpha=alpha)
        png_buffer = io.BytesIO()
        overlay_all.convert("RGB").save(png_buffer, format="PNG")
        st.download_button(
            "Download overlay PNG",
            data=png_buffer.getvalue(),
            file_name=f"{stem}_overlay.png",
            mime="image/png",
        )

    npz_buffer = io.BytesIO()
    np.savez_compressed(
        npz_buffer, **{k.replace(" ", "_"): v for k, v in export_masks_dict.items()}
    )
    st.download_button(
        "Download masks (.npz)",
        data=npz_buffer.getvalue(),
        file_name=f"{stem}_masks.npz",
        mime="application/octet-stream",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        written = export_masks(
            export_masks_dict, data, geometry, Path(tmp_dir) / "masks"
        )
        payloads = [(p.suffix, p.read_bytes()) for p in written]
    for suffix, blob in payloads:
        label = {
            ".vts": "Download masks (.vts - world coordinates, for ParaView)",
            ".vti": "Download masks (.vti - regular grid, for ParaView)",
        }.get(suffix, f"Download masks ({suffix})")
        st.download_button(
            label,
            data=blob,
            file_name=f"{stem}_masks{suffix}",
            mime="application/octet-stream",
            help="Geometry (trace positions, sample interval, delay time) is "
            "read from the SEG-Y headers, so the masks land exactly on the "
            "seismic in ParaView. Threshold by 'label' to isolate features.",
        )


if __name__ == "__main__":
    main()
