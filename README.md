# seismic_prototype

Automatic SAM 3-powered segmentation of 2D `.sgy` seismic sections into five
geological features - **faults, channels, facies, salt bodies, and
horizons** - with no interactive prompting at inference time.

The five text prompts are fixed in [seismic_app/config.py](seismic_app/config.py)
(`SEISMIC_PROMPTS`) and are applied automatically to every tile of every
loaded section.

## Architecture

| Component | File |
|---|---|
| SGY Loader | [seismic_app/sgy_loader.py](seismic_app/sgy_loader.py) |
| Preprocessor (clip/normalize/RGB/tiling) | [seismic_app/preprocessing.py](seismic_app/preprocessing.py) |
| SAM 3 inference engine | [seismic_app/inference.py](seismic_app/inference.py) |
| Mask stitcher | [seismic_app/stitching.py](seismic_app/stitching.py) |
| Viewer / export helpers | [seismic_app/visualization.py](seismic_app/visualization.py) |
| Pipeline orchestration | [seismic_app/pipeline.py](seismic_app/pipeline.py) |
| CLI | [seismic_app/cli.py](seismic_app/cli.py) |
| Streamlit viewer | [app.py](app.py) |

## Setup

```powershell
pip install -r requirements.txt
```

SAM 3 is a **gated** model on Hugging Face. Before running inference:

1. Log in to Hugging Face and accept the license at
   https://huggingface.co/facebook/sam3
2. Authenticate locally:

   ```powershell
   hf auth login
   ```

   (or set the `HF_TOKEN` environment variable). This must be done by you
   directly in a terminal - never share your token in chat.

## Usage

### CLI

```powershell
python -m seismic_app.cli data/1.sgy --out outputs/
python -m seismic_app.cli data/*.sgy --out outputs/
```

Writes `<name>_overlay.png` (color-coded overlay: faults=red, channels=blue,
facies=green, salt=yellow, horizons=cyan) and `<name>_masks.npz`
(per-label boolean masks) to the output directory for each input file.

### Interactive viewer

```powershell
streamlit run app.py
```

Pick a `.sgy` file from `data/`, run segmentation, toggle individual feature
layers, and export the overlay/masks.

## Notes

- This build runs SAM 3 **zero-shot** against the fixed noun-phrase
  vocabulary (no fine-tuning). Quality will improve significantly if the
  checkpoint is fine-tuned on labeled seismic data per the domain-adaptation
  techniques described in the original design guide (frozen vision encoder,
  2.5D input, adapters, BCE+Dice loss, etc.) - that training pipeline is not
  included in this build.
- `.uko` files in `data/` are not SEG-Y and are not read by `sgy_loader.py`.
- If your `.sgy` files are 3D volumes rather than single 2D lines, use
  `seismic_app.sgy_loader.iter_inlines_3d` instead of `load_2d_section`.

