# Fine-tuning SAM 3 for seismic facies

This folder is self-contained: it does not import from or get imported by
`seismic_app/`, except to reuse two small, already-tested pieces of
preprocessing code (`normalize_to_uint8`, `inline_to_rgb_25d`) so training
images are built with the *exact* same pipeline the app uses at inference
time.

## 0. What data this actually covers

`C:\Users\...\Data\trainingData_256_7_28_26` has two folders, `Seis80/` and
`ImpMask80/`, each with 84 files named `0.dat`...`83.dat` (matched by
number). Inspecting the raw bytes (see chat history / re-derivable with the
snippet below) showed:

- `Seis80/<id>.dat` - a `(256, 256, 256)` **float64** seismic amplitude
  cube, already roughly zero-mean and in a `[-5, 5]`-ish range.
- `ImpMask80/<id>.dat` - a `(256, 256, 256, 20)` **float64** cube of
  strictly-`{0, 1}` values, **bins last**. This axis order was only
  confirmed by trial and error: reshaping with bins *first* (the more
  "natural"-looking guess) produces masks that are pure vertical-stripe
  noise with zero correlation to the seismic structure at the same slice;
  bins *last* produces a smooth mask that follows the same folds visible
  in the seismic section - see the slice comparison images generated
  while debugging this, reproducible with the snippet below. The 20
  channels are **not** one-hot (a voxel can be "on" in several channels at
  once, average ~3 of 20), and the per-channel "on" fraction rises then
  falls smoothly across the 20 channels - consistent with 20 overlapping,
  ordered low->high **acoustic-impedance bins**, not 20 unrelated
  geological classes.

```python
import numpy as np
seis = np.fromfile("Seis80/0.dat", dtype="<f8").reshape(256, 256, 256)
mask = np.fromfile("ImpMask80/0.dat", dtype="<f8").reshape(256, 256, 256, 20)
```

**Decision made with the user** (this dataset only covers one of the app's
five noun_phrases):

- Target noun_phrase: **`"seismic facies"`** only. `fault`, `channel`,
  `salt body`, `horizon` stay zero-shot until matching labeled data shows up.
- The 20 impedance bins collapse to one binary mask via **union** (voxel is
  "facies" if it's in ANY of the 20 bins). Measured on the sample cube this
  covers ~75% of the volume - i.e. the resulting mask is closer to
  "classified impedance region vs. background/no-data" than a sparse
  object mask. If your intended target is something narrower, rerun
  `prepare_dataset.py --collapse bin-range --min-bin ... --max-bin ...`
  and adjust the pipeline below - everything downstream is unaffected by
  which collapse you pick.

## 1. Environment setup (one-time)

Verified against the actual `facebookresearch/sam3` repo's own
Installation section and README_TRAIN.md - this is a **separate** Python
environment from the one this app's `requirements.txt` sets up, because
the sam3 repo pins its own torch/CUDA versions:

```powershell
conda create -n sam3-finetune python=3.12
conda activate sam3-finetune

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e ".[train]"

# Optional, faster attention (needs a matching CUDA build):
pip install einops ninja
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128

# Accept the license at https://huggingface.co/facebook/sam3, then:
hf auth login
```

Then, from *this* app's repo root, install the extra tooling used by the
scripts in this folder (pycocotools, tensorboard) into whichever
environment you run `prepare_dataset.py` / `verify_dataset.py` /
`validate_checkpoint.py` from (these three don't need the sam3 repo at
all - only `transformers`, which is already in the main `requirements.txt`):

```powershell
pip install -r finetuning/requirements-finetune.txt
```

## 2. Prepare the dataset (Step 1)

```powershell
python -m finetuning.prepare_dataset `
    --seis-dir "C:\Users\umair_tariq\Downloads\Downloads\Data\trainingData_256_7_28_26\Seis80" `
    --mask-dir "C:\Users\umair_tariq\Downloads\Downloads\Data\trainingData_256_7_28_26\ImpMask80" `
    --out-dir finetune_data `
    --slice-stride 2
```

This slices every cube along the inline axis (axis 0 - same convention as
`seismic_app/preprocessing.inline_to_rgb_25d`), builds a 2.5D RGB PNG per
slice (previous/current/next inline in R/G/B, exactly like the app does
for real 3D volumes), collapses the mask, and writes:

```
finetune_data/
  images/{train,val}/cube###_ilNNNN.png
  masks/{train,val}/cube###_ilNNNN.png   # human-viewable, not used for training
  train.json                             # COCO json, noun_phrase="seismic facies"
  val.json
```

The 85/15 split is by **cube id**, not by slice, so validation slices never
come from a cube the model trained on. `--slice-stride 2` keeps the dataset
to a manageable size (~half the 256 slices/cube); use `--slice-stride 1`
for the full set once the pipeline is validated end-to-end.

## 3. Verify the dataset (Step 2)

```powershell
python -m finetuning.verify_dataset finetune_data
```

Checks image files exist and match their recorded size, every
annotation's `noun_phrase` matches its category (catches a drifted prompt
string before it silently breaks training), reports positive/negative
image balance, and spot-checks that decoded RLE masks match their
recorded area.

## 4. Training config (Steps 4 & 7 - freezing strategy)

[`configs/seismic_facies_phase1.yaml`](configs/seismic_facies_phase1.yaml)
is adapted line-by-line from sam3's own
`sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml`
(fetched and checked directly), pointed at `finetune_data/{train,val}.json`
instead of a Roboflow dataset, with segmentation turned on
(`enable_segmentation: true`) and a BCE(focal)+Dice mask loss.

**Freezing mechanism** (verified against `sam3/model_builder.py`): there is
no `freeze_encoder=True` kwarg on `build_sam3_image_model`. Freezing is
done the same way this repo already assigns per-parameter-group learning
rates - Phase 1 sets `lr_vision_backbone: 0.0` and
`lr_language_backbone: 0.0`, so only the DETR decoder/segmentation head
trains.

- **Phase 1** (this config): decoder only, vision + text encoder frozen.
- **Phase 2**: unfreeze the last 8 of the ViT trunk's 32 blocks (blocks
  24-31; depth=32 is hardcoded in `model_builder._create_vit_backbone`).
  See [`configs/seismic_facies_phase2_overrides.yaml`](configs/seismic_facies_phase2_overrides.yaml)
  for the exact diff to apply on top of a copy of the Phase 1 file.
- **Phase 3**: same pattern, unfreeze the last 16 of 32 blocks (16-31) -
  described at the bottom of the phase 2 file, not written out in full.

Before trusting the block-name pattern
(`backbone.vision_backbone.trunk.blocks.<N>.*`), confirm it against your
checkout:

```powershell
grep -n "self.blocks" sam3\model\vitdet.py
```

Augmentation: horizontal flip only
(`sam3.train.transforms.basic_for_api.RandomHorizontalFlip`, a real class
in that file). There is no vertical-flip or 90-degree-rotation transform
class in the repo to begin with, which is convenient here since either
would invert this app's "time down, traces across" orientation convention.

Every `_target_` path in the yaml files was checked against the live
`facebookresearch/sam3` and `huggingface/transformers` source; a few
dataset-loader wiring details (`COCO_FROM_JSON` used for a *train* split,
not just val) are marked `VERIFY` in the config comments because I could
only confirm them being used for the *val* split in the reference config -
double check `sam3/train/data/coco_json_loaders.py` if the trainer errors
on the train dataloader.

## 5. Launch training (Step 5)

```powershell
cd sam3
# Single GPU
python sam3/train/train.py -c <path_to>/seismic_facies_phase1.yaml --use-cluster 0 --num-gpus 1

# Multiple GPUs on one machine
python sam3/train/train.py -c <path_to>/seismic_facies_phase1.yaml --use-cluster 0 --num-gpus 4
```

This repo's own `train.py` spawns the worker processes itself via
`--num-gpus`/`--num-nodes` - there's no separate `torchrun` invocation to
run by hand for local training (that's only relevant for
`--use-cluster 1` / SLURM, which this README doesn't cover since it needs
your cluster's partition/account names).

Quick experiments: override any key from the CLI, e.g.
`... -c seismic_facies_phase1.yaml scratch.train_batch_size=4 trainer.max_epochs=5`.

## 6. Monitor with TensorBoard (Step 6)

```powershell
tensorboard --logdir <experiment_log_dir>/tensorboard
```

Watch `train/*_loss` drop over the first 10-15 epochs; if it plateaus
immediately or NaNs, see [Common errors](#7-common-errors--fixes) below.
There is no built-in per-class mIoU meter verified in this repo for
segmentation training, so track val quality with
[`validate_checkpoint.py`](#8-validate-a-checkpoint-step-9) run against
saved checkpoints instead of a TensorBoard scalar.

## 7. Checkpoint management (Step 8)

Verified against `sam3/train/trainer.py`'s `save_checkpoint`: every epoch
writes a rolling `checkpoints/checkpoint.pt`, and `checkpoint.save_freq: N`
(N>0) additionally keeps a numbered `checkpoints/checkpoint_<epoch>.pt`
every N epochs. There is no separate "best_checkpoint.pt" produced unless
you configure a `meters:` block and list its key in
`checkpoint.save_best_meters` - not set up here (no verified segmentation
mIoU meter class in this repo), so pick the epoch to use for Phase 2 /
the app by eyeballing TensorBoard val loss, or by running
`validate_checkpoint.py` (below) against a few candidate
`checkpoint_<epoch>.pt` files.

To resume a crashed run, set `checkpoint.resume_from: <path to
checkpoint.pt>` in the yaml (the trainer copies it into
`checkpoint.save_dir/checkpoint.pt` and continues from its stored epoch).

## 8. Validate a checkpoint (Step 9)

The saved trainer checkpoint is in the *original* sam3 repo's own format,
not something `transformers.Sam3Model` can load directly. Convert it
first, using HF transformers' own conversion script (not shipped in the
pip package - fetch it once from source, it's a single file):

```powershell
curl -o convert_sam3_to_hf.py https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/sam3/convert_sam3_to_hf.py

python convert_sam3_to_hf.py `
    --checkpoint_path <experiment_log_dir>/checkpoints/checkpoint.pt `
    --output_path finetuned_checkpoints/seismic_facies_phase1
```

This writes an HF-format directory (`config.json` + safetensors) and, at
the end, does its own smoke-test reload with `Sam3Model.from_pretrained`.

Then compute mean IoU on the held-out val split, through this app's own
inference code path:

```powershell
python -m finetuning.validate_checkpoint `
    --dataset-root finetune_data `
    --checkpoint finetuned_checkpoints/seismic_facies_phase1

# Baseline for comparison (zero-shot, no fine-tuning):
python -m finetuning.validate_checkpoint --dataset-root finetune_data --checkpoint facebook/sam3
```

## 9. Load into the application (Step 10)

Exactly one line changes, because `Sam3Model.from_pretrained()` /
`Sam3Processor.from_pretrained()` accept a local directory just as
happily as a Hugging Face Hub repo id - in
[`seismic_app/config.py`](../seismic_app/config.py):

```python
SAM_FAMILIES["sam3"]["checkpoint"] = r"C:\...\finetuned_checkpoints\seismic_facies_phase1"
```

or, without editing code, pass it directly: `--checkpoint
C:\...\finetuned_checkpoints\seismic_facies_phase1` to `seismic_app.cli`,
or the equivalent field in the Streamlit/React UI. Nothing else in
`seismic_app/` changes - tiling, stitching, VTK export, and the point
picker are all checkpoint-agnostic.

## 10. Common errors & fixes (Step 11)

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss is `NaN` after a few steps | LR too high once a frozen block is unfrozen (Phase 2/3), or bf16 amp overflow | Lower `lr_vision_backbone`, or switch `optim.amp.amp_dtype` to `float16` with the gradient scaler (already enabled) |
| `val` loss/mIoU stuck at ~0 for many epochs | `noun_phrase` mismatch between train/val JSON and the checkpoint's text prompt at inference, or all `--collapse union` masks came out empty for a wrong axis assumption | Re-run `verify_dataset.py`; open a few `masks/train/*.png` by eye to confirm they aren't blank |
| `CUDA out of memory` | `scratch.resolution=1008` at `train_batch_size>1` is large for a single consumer GPU | Drop `train_batch_size` to 1, raise `gradient_accumulation_steps`, or drop `resolution` |
| Predictions are all-background at inference | Forgot to convert the trainer checkpoint before pointing the app at it (raw `.pt` isn't the same format `Sam3Model.from_pretrained` expects) | Run the `convert_sam3_to_hf.py` step above first |
| Fine-tuned checkpoint looks *worse* than zero-shot after a few epochs | Overfitting to only 71 training cubes, or vision backbone unfrozen too early | Go back to Phase 1 only (frozen encoders) for longer, or reduce `--slice-stride` in `prepare_dataset.py` for more (correlated but still useful) training slices |
| `checkpoint_path errors` (file not found / wrong keys) loading into transformers | Pointed `Sam3Model.from_pretrained` at the raw sam3-repo `checkpoint.pt` instead of the converted HF directory, or a stale/partial `.tmp` file from an interrupted save | Point at the `convert_sam3_to_hf.py --output_path` directory, and make sure training wasn't killed mid-checkpoint-write (the trainer's `_save_checkpoint` uses a `.tmp` + atomic rename, so a finished `checkpoint.pt` is always valid) |

## Files in this folder

| File | Purpose |
|---|---|
| [`prepare_dataset.py`](prepare_dataset.py) | Cubes -> 2D PNG slices + COCO json (train/val, 85/15 by cube) |
| [`verify_dataset.py`](verify_dataset.py) | Sanity-check the COCO json before training |
| [`configs/seismic_facies_phase1.yaml`](configs/seismic_facies_phase1.yaml) | Hydra training config, decoder-only frozen encoders |
| [`configs/seismic_facies_phase2_overrides.yaml`](configs/seismic_facies_phase2_overrides.yaml) | Diff to apply for Phase 2 (unfreeze last 8/32 ViT blocks) |
| [`validate_checkpoint.py`](validate_checkpoint.py) | Per-image + mean IoU on val split, via this app's own inference code |
| [`requirements-finetune.txt`](requirements-finetune.txt) | Extra deps (pycocotools, tensorboard) for the scripts above |
