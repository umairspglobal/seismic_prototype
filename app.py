"""Viewer / UI component (guide section 6).

Run with:
    streamlit run app.py

Loads a .sgy file, runs the automatic 5-feature SAM 3 segmentation, and
lets the user toggle individual feature layers on/off and export results.
No interactive prompting of SAM 3 is required - the five noun-phrase
prompts are fixed in seismic_app/config.py and applied automatically.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st

from seismic_app import config
from seismic_app.inference import Sam3SeismicSegmenter
from seismic_app.pipeline import run_on_file
from seismic_app.visualization import overlay_masks

DATA_DIR = Path("data")


@st.cache_resource(show_spinner="Loading SAM 3 checkpoint (first run only)...")
def load_segmenter(checkpoint: str, device: str | None) -> Sam3SeismicSegmenter:
    return Sam3SeismicSegmenter(checkpoint=checkpoint, device=device)


def list_sgy_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.sgy"))


def main() -> None:
    st.set_page_config(page_title="SAM 3 Seismic Segmentation", layout="wide")
    st.title("SAM 3 Seismic Segmentation")
    st.caption(
        "Automatic detection of faults, channels, facies, salt bodies, and "
        "horizons in 2D .sgy sections - no interactive prompting required."
    )

    with st.sidebar:
        st.header("Settings")
        sgy_files = list_sgy_files()
        options = [str(p) for p in sgy_files] or ["(no .sgy files found in data/)"]
        selected = st.selectbox("Seismic section", options)
        checkpoint = st.text_input("SAM 3 checkpoint", value=config.DEFAULT_CHECKPOINT)
        device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
        threshold = st.slider(
            "Mask probability threshold", 0.0, 1.0, config.MASK_THRESHOLD, 0.05
        )
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
        run_clicked = st.button("Run segmentation", type="primary")

        st.divider()
        st.subheader("Layers")
        layer_toggles = {
            style.noun_phrase: st.checkbox(style.noun_phrase, value=True)
            for style in config.LABEL_STYLES
        }

    if not sgy_files:
        st.warning(f"No .sgy files found in {DATA_DIR}/. Add some and reload.")
        return

    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.result_path = None

    if run_clicked:
        segmenter = load_segmenter(checkpoint, None if device == "auto" else device)
        with st.spinner(f"Segmenting {selected} ..."):
            result = run_on_file(selected, segmenter, threshold=threshold)
        st.session_state.result = result
        st.session_state.result_path = selected

    result = st.session_state.result
    if result is None:
        st.info("Choose a section and click **Run segmentation** to begin.")
        return

    active_layers = [name for name, on in layer_toggles.items() if on]
    overlay = overlay_masks(result.rgb, result.masks, alpha=alpha, only=active_layers)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original section")
        st.image(result.rgb, use_container_width=True)
    with col2:
        st.subheader("Segmentation overlay")
        st.image(np.array(overlay.convert("RGB")), use_container_width=True)

    st.subheader("Detected coverage per feature")
    cols = st.columns(len(config.LABEL_STYLES))
    for col, style in zip(cols, config.LABEL_STYLES):
        mask = result.masks.get(style.noun_phrase)
        pct = 100.0 * mask.mean() if mask is not None else 0.0
        col.metric(style.noun_phrase, f"{pct:.2f}%")

    st.subheader("Export")
    png_buffer = io.BytesIO()
    overlay.convert("RGB").save(png_buffer, format="PNG")
    st.download_button(
        "Download overlay PNG",
        data=png_buffer.getvalue(),
        file_name=f"{Path(st.session_state.result_path).stem}_overlay.png",
        mime="image/png",
    )

    npz_buffer = io.BytesIO()
    np.savez_compressed(
        npz_buffer, **{k.replace(" ", "_"): v for k, v in result.masks.items()}
    )
    st.download_button(
        "Download masks (.npz)",
        data=npz_buffer.getvalue(),
        file_name=f"{Path(st.session_state.result_path).stem}_masks.npz",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
