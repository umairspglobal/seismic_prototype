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

### React interactive client (this branch)

The smooth, SAM 2 demo-style interface: a React/TypeScript/Vite frontend
talking to a persistent FastAPI inference service. Models, image
embeddings, and video sessions stay resident between requests, so a
point click round-trips in tens of milliseconds and propagation streams
each tracked slice to the browser as it completes (scrub while it runs).

Prompting follows the SAM 2 demo conventions: left click adds an object
point, right click adds a background point, and multiple objects can be
tracked at once - add objects in the sidebar, click points for each, and
propagate. Objects and their points persist while you scrub: move to any
slice, add or undo refinement clicks for an object there, and
re-propagate. After the first propagate, +/− clicks on any slice edit
that slice's tracked mask immediately (a − click carves the blob you
clicked, a + click grows it), then **Re-propagate** spreads those edits
through the volume. **Download tracked volume (.vti)** writes a ParaView
ImageData file you can open directly — threshold the `label` array
(object 1 = 6, object 2 = 7, …). Objects share one tracker session up
to the GPU's free VRAM;
anything beyond that is queued into follow-up waves automatically. Each
slice streams to the browser as a single combined multi-color mask.

The tracker models are loaded in the background as soon as the inference
server starts. The React client shows a loading overlay until the point
tracker is on the GPU, then encodes the active slice so the first click
is already warm.

Start the inference server (models begin loading immediately):

```powershell
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Start the frontend dev server in a second terminal:

```powershell
cd frontend
npm install   # first time only (requires Node.js LTS)
npm run dev
```

Then open http://localhost:5173. Or run both at once:

```powershell
.\start_react_app.ps1
```

Verify the API end-to-end (server must be running):

```powershell
python benchmarks/api_smoke_test.py --propagate
```

### Streamlit viewer (legacy)

```powershell
streamlit run app.py
```

- Pick a `.sgy` file; the sidebar shows the geometry read from its
  headers (type, sample interval, trace spacing, CDP range).
- **Automatic tab**: run the five fixed text prompts, toggle layers.
- **Point picking tab**: the active slice is encoded before clicking, then
  positive/negative clicks update the mask from cached features. The last
  click-to-preview and decoder timings are shown below the picker.
- For 3D volumes the sidebar gains a slice navigator
  (inline / crossline / time slice); picking operates on the active slice.
- Live masks stay lightweight until **Prepare export files** is clicked.
  This avoids rebuilding compressed NPZ and ParaView payloads on every pick.

### CUDA point and propagation benchmark

The non-Streamlit benchmark harness measures interactive point inference on
2D or 3D SEG-Y data and volume propagation on 3D data:

```powershell
python benchmarks/point_propagation_benchmark.py data/SEGY0000.sgy `
  --checkpoint facebook/sam3 `
  --device cuda `
  --axis inline `
  --anchor 10 `
  --warm-runs 5
```

`--anchor` is zero-based and defaults to the middle slice. `--axis` accepts
`inline`, `crossline`, or `time`. Use `--mode point`, `--mode propagation`,
or the default `--mode both`. For a 2D line, point inference uses the full
section and propagation is reported as skipped.

The harness writes a JSON report to standard output. Model loading is timed
separately from inference. Point results include the first (cold,
vision-encoder) call, every cached warm call, warm p50/p95, and stage timings.
Propagation reports frame preparation, video-session setup, tracker inference,
cold/warm total time, p50/p95, and FPS. Use `--max-frames 15` for a quick
centered sample, `--float32` for the unoptimized precision comparison, or
`--compile` to evaluate `torch.compile` on a supported runtime. On CUDA, peak
allocated and reserved VRAM are reported after model load.

For automatic device selection, pass `--device auto`. It selects CUDA when
available and otherwise runs on CPU with a note in the report. An explicit
CUDA request also falls back to CPU when this PyTorch installation has no
CUDA support. The checkpoint must already be available locally or accessible
through an authenticated Hugging Face session; running the benchmark may
otherwise download it. For the most comparable results, close other GPU
workloads and keep the SEG-Y file, checkpoint, device, axis, anchor, and warm
run count unchanged.

## Notes

- This build runs SAM 3 **zero-shot** (no seismic fine-tuning). Text-prompt
  quality against natural-image-trained weights will vary per concept; the
  interactive point mode is typically much more reliable, since you tell
  the model exactly where the object is.
- `.uko` files in `data/` are not SEG-Y and are not read.
