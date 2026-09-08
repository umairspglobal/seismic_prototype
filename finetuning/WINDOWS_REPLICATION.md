# SAM 3 fine-tuning on Windows — replication report

Verified working on **2026-09-08**: training reached step 40+ of epoch 0 with the
loss average falling 182 → 69 at ~2.7 s/step and ~7 GB VRAM (batch size 1,
resolution 1008).

This document is the *Windows-specific* companion to `finetuning/README.md`.
The README covers dataset preparation, the training config, checkpoint
conversion, and pointing the app at a fine-tune. This file covers everything
that had to be fixed to make training actually run on a Windows machine, so
you can replicate the working setup on a different PC.

## Machine this was verified on

| Item | Value |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 3080 Laptop, 16 GB (driver 596.52) |
| Python | 3.12.0 (plain python.org install, **not** conda) |
| PyTorch | **2.10.0+cu128** (see "Critical version pins" below) |
| sam3 repo | `https://github.com/facebookresearch/sam3.git` @ commit `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` |

## Critical version pins

1. **`torch==2.10.0+cu128` — do not use 2.11.** On torch 2.11 the forward pass
   dies with `CUDA error: an illegal memory access was encountered` (reported
   from `F.conv2d` in the FPN neck, but it is an async CUDA fault). Downgrading
   to the 2.10.0 cu128 wheel fixed it outright.
2. **`setuptools<81`** — newer setuptools removed `pkg_resources`, which sam3's
   training code imports.
3. **`numpy 1.26.x`** — already satisfied by the main app's requirements; don't
   let anything upgrade it to numpy 2.

## Step-by-step setup on a new PC

All commands are PowerShell, run from the repo root
(`...\seismic_prototype`) unless noted.

### 1. Base app environment

Follow the main repo README first (install `requirements.txt`, build the
frontend, etc.). The training setup below goes into the same Python 3.12
environment we used, but a dedicated venv works too.

### 2. Clone sam3 and install it with training extras

```powershell
git clone https://github.com/facebookresearch/sam3.git
git -C sam3 checkout 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
python -m pip install "torch==2.10.0" torchvision --index-url https://download.pytorch.org/whl/cu128
cd sam3
python -m pip install -e ".[train]"
cd ..
python -m pip install "setuptools<81"
python -m pip install einops timm ftfy regex iopath hydra-core submitit tensorboard zstandard scipy torchmetrics fvcore fairscale scikit-image scikit-learn pycocotools
```

Install torch *before* `pip install -e ".[train]"` so pip keeps the pinned
CUDA build instead of resolving its own.

Sanity check:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.10.0+cu128 True
```

### 3. Apply the Windows patch to the sam3 source

The sam3 checkout needs local modifications (it is gitignored by this repo, so
they are captured in `finetuning/patches/sam3_windows_fixes.patch`):

```powershell
cd sam3
git apply ..\finetuning\patches\sam3_windows_fixes.patch
cd ..
```

What the patch does (all changes are `sys.platform == "win32"`-guarded or
behavior-preserving on Linux):

| Area | Change | Why |
|---|---|---|
| `train/train.py` | `MASTER_ADDR=127.0.0.1`, `USE_LIBUV=0` | `localhost` fails gloo device resolution on Windows |
| `train/utils/train_utils.py` | if `init_process_group` raises and `WORLD_SIZE=1`, continue without a process group | torch 2.10's gloo backend cannot create a device on Windows at all (`makeDeviceForHostname(): unsupported gloo device`); single-GPU training doesn't need one |
| `train/trainer.py` | allow uninitialized torch.distributed when `WORLD_SIZE=1`; skip the DDP wrap for single process; Phase-1 backbone freeze helper (`requires_grad=False` + eval + `no_grad`); disable activation checkpointing of the seg head on Windows | crash avoidance + the freeze strategy the config relies on |
| `train/data/torch_dataset.py` | fall back from `DistributedSampler` to plain `DataLoader(shuffle=...)` when no process group exists | `DistributedSampler` requires an initialized process group |
| `train/loss/sam3_loss.py`, `train/loss/loss_fns.py` | only `all_reduce` when world size > 1 | gloo `all_reduce` on CUDA tensors crashes natively on Windows |
| `train/matcher.py` | `np.nan_to_num` on the Hungarian cost matrix | fp16 AMP can produce NaN costs early in training; `linear_sum_assignment` hard-fails on them |
| `model/vitdet.py`, `sam/transformer.py`, `model/model_misc.py`, `model/decoder.py`, `model/vl_combiner.py`, `model/edt.py`, `perflib/fused.py` | force the MATH SDPA backend / disable flash + mem-efficient attention on Windows | flash/mem-efficient SDPA kernels caused `CUDA illegal memory access` crashes |
| `train/optim/optimizer.py` | minor Windows-compat tweaks | part of the same crash triage |

### 4. Put the training config where Hydra can find it

Hydra resolves config names against `sam3/sam3/train/configs/`, not arbitrary
relative paths:

```powershell
Copy-Item finetuning\configs\seismic_facies_phase1.yaml sam3\sam3\train\configs\
```

### 5. Edit the absolute paths in the config

`sam3\sam3\train\configs\seismic_facies_phase1.yaml` has a `paths:` block at
the top with four machine-specific absolute paths. Update all of them for the
new PC:

- `dataset_root` — output of `prepare_dataset.py` (step 6)
- `experiment_log_dir` — where logs/checkpoints will be written
- `bpe_path` — `<repo>/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz` (ships with the sam3 clone)
- `init_checkpoint` — the pretrained `sam3.pt`. Get it once with:

  ```powershell
  python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('facebook/sam3', 'sam3.pt'))"
  ```

  and paste the printed path (forward slashes are fine).

### 6. Prepare the dataset

Same as `finetuning/README.md`:

```powershell
python -m finetuning.prepare_dataset `
  --seis-dir "<path>\data\Seis80" `
  --mask-dir "<path>\data\ImpMask80" `
  --out-dir finetune_data `
  --slice-stride 8 `
  --max-cubes 36
```

### 7. Launch training

```powershell
cd sam3
python sam3/train/train.py -c configs/seismic_facies_phase1.yaml --use-cluster 0 --num-gpus 1
```

Note the `-c` path is just `configs/seismic_facies_phase1.yaml` (relative to
the Hydra search path, i.e. the file copied in step 4).

### 8. What a healthy start looks like

- A warning that torch.distributed init failed and it is continuing without a
  process group (expected on Windows — that is our patch).
- `Total parameters 840 M / Trainable parameters 32.7 M` (Phase-1 freeze OK).
- `Raw dataset length = 992` (train) and `160` (val).
- Progress lines every 10 steps. First step is slow (~30 s, CUDA warmup), then
  ~2.7 s/step on this GPU:

  ```text
  Train Epoch: [0][  0/992] | ... | Losses/train_all_loss: 1.82e+02 (1.82e+02)
  Train Epoch: [0][ 10/992] | ... | Losses/train_all_loss: 8.06e+01 (1.51e+02)
  Train Epoch: [0][ 40/992] | ... | Losses/train_all_loss: 5.74e+01 (6.89e+01)
  ```

  The loss average should fall steadily. One epoch ≈ 45 min at this speed.

Checkpoints land in `<experiment_log_dir>/checkpoints/checkpoint.pt` (rolling,
written after every epoch). Convert them with `convert_sam3_to_hf.py` into a
folder under `finetuned_checkpoints/` (see `finetuning/README.md`, "Convert
the checkpoint"); the React app's "Fine-tuned checkpoint" dropdown picks up
every converted folder there automatically.

## Every error hit on this machine, in order, with the fix

| # | Error | Cause | Fix |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'submitit'` | training extras not installed | `pip install -e ".[train]"` in `sam3/` |
| 2 | `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools ≥ 81 removed it | `pip install "setuptools<81"` |
| 3 | `ModuleNotFoundError: No module named 'einops'` (then timm, ftfy, …) | extras list incomplete | the big `pip install` line in step 2 |
| 4 | `hydra.errors.MissingConfigException` for the yaml | Hydra only searches `sam3/sam3/train/configs/` | copy the config there (step 4) |
| 5 | `RecursionError` in `trainer.py` `_stay_eval` | freeze hook re-entered itself via the patched `.train()` | patched: call `torch.nn.Module.train(mod, False)` directly |
| 6 | `ConfigAttributeError: Missing key info ... full_key: logging.info` | `logging` name shadowed by the Hydra config node inside `Trainer.__init__` | patched: use `__import__("logging").info` there |
| 7 | `ValueError: matrix contains invalid numeric entries` in `linear_sum_assignment` | NaN costs from fp16 AMP early steps | patched: `np.nan_to_num` in `matcher.py` |
| 8 | `CUDA error: an illegal memory access was encountered` (several call sites) | flash / mem-efficient SDPA kernels + act-checkpointing on Windows; residual crashes were a torch 2.11 regression | patched: MATH SDPA backend + no act-ckpt on Windows; **downgrade to torch 2.10.0+cu128** |
| 9 | `RuntimeError: makeDeviceForHostname(): unsupported gloo device` | torch 2.10's gloo cannot create a device on Windows (any address) | patched: skip the process group entirely for `WORLD_SIZE=1` (train_utils/trainer/torch_dataset changes) |

Errors 5–9 are all contained in `finetuning/patches/sam3_windows_fixes.patch`,
so on a new PC you only ever see them if you forget step 3.

## Exact package versions verified working

```text
torch==2.10.0+cu128        torchvision==0.25.0+cu128   numpy==1.26.4
setuptools==80.10.2        einops==0.8.2               submitit==1.5.4
hydra-core==1.3.6          tensorboard==2.21.0         zstandard==0.25.0
scipy==1.17.1              torchmetrics==1.9.0         fvcore==0.1.5.post20221221
fairscale==0.4.13          scikit-image==0.26.0        scikit-learn==1.9.0
pycocotools==2.0.11        timm==1.0.29                ftfy==6.1.1
iopath==0.1.10             transformers==5.15.1        huggingface_hub==1.28.0
```
