# seismic_prototype

Geometry-aware SAM 3 segmentation of `.sgy` seismic data.

Two modes:

- **Automatic (text prompts)** - zero-shot detection of five geological
  features (**faults, channels, facies, salt bodies, horizons**) using the
  fixed noun-phrase vocabulary in [seismic_app/config.py](seismic_app/config.py).
- **Interactive (point picking)** - click positive/negative points directly
  on the section in the viewer; SAM 3's tracker head segments the object
  you indicated. Picks are reported in physical coordinates (CDP number
  and time in ms) read from the SEG-Y headers.

## Geometry handling

SEG-Y files are not photographs: their samples live on physical axes
(trace position / inline / crossline, and time down the trace). The
pipeline extracts that geometry once at load time
([seismic_app/geometry.py](seismic_app/geometry.py)) and threads it
through every stage:

- **Auto-detection**: files whose inline/crossline headers form a sorted
  3D volume are loaded as cubes; everything else (including all files
  currently in `data/`, which are 2D crooked lines) is loaded as a 2D
  section.
- **Orientation**: sections are held as `(n_samples, n_traces)` - time
  down, traces across - so SAM sees horizons as horizontal features and
  the display matches standard seismic convention.
- **Header values used**: sample interval (cross-checked between trace
  and binary headers, since some files carry a bogus binary-header
  value), recording delay, CDP numbers, and per-trace world coordinates
  (from which the real trace spacing is measured - e.g. `data/1.sgy` is
  ~68 m, not a guessed 25 m).
- **2.5D RGB**: for 3D volumes, each inline is presented to SAM with the
  previous/current/next inlines in the R/G/B channels. 2D lines use
  channel duplication (they have no neighboring sections).

## ParaView export

For 2D lines two files are written:

- `<name>_masks.vts` (**recommended**) - a StructuredGrid placing every
  trace at its world (x, y) with time as negative-down z, mirroring what
  ParaView's own SEG-Y reader builds. The raw `amplitude` is included
  next to the `label` array, so you can verify alignment without loading
  the .sgy at all.
- `<name>_masks.vti` - a flat panel in (distance-along-line, time) space.

3D volumes export a single `.vti` with inline/crossline bin spacings and
the sample interval as axis spacings.

Label IDs in the exported `label` array: 0=background, 1=fault,
2=channel, 3=facies, 4=salt, 5=horizon; interactively picked masks get
IDs from 6 upward. Use ParaView's Threshold filter per label ID.

## Architecture

| Component | File |
|---|---|
| Geometry extraction (axes, spacing, coordinates) | [seismic_app/geometry.py](seismic_app/geometry.py) |
| SGY loader (2D/3D auto-detect, orientation) | [seismic_app/sgy_loader.py](seismic_app/sgy_loader.py) |
| Preprocessor (clip/normalize/RGB/2.5D/tiling) | [seismic_app/preprocessing.py](seismic_app/preprocessing.py) |
| SAM 3 inference (text prompts + point prompts) | [seismic_app/inference.py](seismic_app/inference.py) |
| Mask stitcher | [seismic_app/stitching.py](seismic_app/stitching.py) |
| Viewer / overlay helpers | [seismic_app/visualization.py](seismic_app/visualization.py) |
| VTK export (.vts/.vti with real geometry) | [seismic_app/vtk_export.py](seismic_app/vtk_export.py) |
| Pipeline orchestration | [seismic_app/pipeline.py](seismic_app/pipeline.py) |
| CLI | [seismic_app/cli.py](seismic_app/cli.py) |
| Streamlit viewer (tabs: automatic + point picking) | [app.py](app.py) |

## Setup

```powershell
pip install -r requirements.txt
```

Note: SAM 3 requires a transformers release that includes it
(`transformers>=4.58`); the requirements file enforces this.

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

Writes, per input file: `<name>_overlay.png`, `<name>_masks.npz`,
`<name>_masks.vts` (world coordinates) and `<name>_masks.vti`. Trace
spacing and sample interval are read from the SEG-Y headers;
`--trace-spacing` / `--sample-interval` are optional overrides.

### Interactive viewer

```powershell
streamlit run app.py
```

- Pick a `.sgy` file; the sidebar shows the geometry read from its
  headers (type, sample interval, trace spacing, CDP range).
- **Automatic tab**: run the five fixed text prompts, toggle layers.
- **Point picking tab**: choose positive/negative, click points on the
  section (each pick is echoed as CDP + time), press *Segment from
  points*, then *Add to layers* with a name of your choosing. Interactive
  masks are exported alongside the automatic ones.
- For 3D volumes the sidebar gains a slice navigator
  (inline / crossline / time slice); picking operates on the active slice.

## Notes

- This build runs SAM 3 **zero-shot** (no seismic fine-tuning). Text-prompt
  quality against natural-image-trained weights will vary per concept; the
  interactive point mode is typically much more reliable, since you tell
  the model exactly where the object is.
- `.uko` files in `data/` are not SEG-Y and are not read.
