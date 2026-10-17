# Seismic Synthetic Generator

Builds faulted seismic training data for the SAM 3 Quick Fine-Tuner, in exactly the
layout of `training_10_7_26`: one folder per array, one raw little-endian float64
`224 x 224` file per sample id. `1.dat` in the repository root is `Fault75/1.dat`
of that dataset (block 0, slice 1) and is what a generated `Fault75/<id>.dat` looks like.

```powershell
start_synthetic_generator.cmd                      # GUI (or: python -m synthetic_generator)
python -m synthetic_generator.cli --help           # command line
```

## Quick start

1. Start the GUI. The right side shows a live preview of the block that will be written.
2. Pick a preset (top right) and adjust any setting; the preview refreshes as you type.
   Preview block *k* is exactly dataset block *k* for the same seed.
3. Choose the output folder and the number of blocks in **Generate dataset**, then
   press **Generate dataset**. A block is one geological model, written as
   6 slices x 4 bandpass replicas = 24 ids.
4. **Send to Quick Fine-Tuner** writes the fine-tuner's settings
   (`Seis75` + `Fault75` for faults, or `Seis75` + layers from `TauAtT75` for facies)
   and opens it with this folder selected.

Disk use: 15 arrays x 24 ids x 401 KB = about 145 MB per block (float64). 400 blocks
(9,600 ids, the size of training_10_7_26) take about 58 GB; the GUI shows the size, free
space and an estimate before it starts. Choose fewer arrays (Dataset tab) or float32 to
save space - the fault fine-tuner only needs `Seis` and `Fault`.

## Output layout

| Folder (suffix `75`) | Content | Differs between replicas |
|---|---|---|
| `Seis75` | seismic amplitude, mean 0, std 1 | yes |
| `Fault75` | fault stroke mask, 1 = fault, 0 = background | no |
| `TauAtT75` | relative geologic time (tau) | no |
| `dipXscale{S}_75` | `tau[y, x] - tau[y, x + S]` | no |
| `wtXscale{S}_75` | scale weight (loss mask), 0 / 1 | no |
| `wtXStraddleScale{S}_75` | straddle, 0 / 1 | no |
| `wtXHaloScale{S}_75` | halo, 0..1 | no |

With the default scales S = 12, 36, 60 that is fifteen arrays per id. Ids are
replica-major, like the reference:

```
id = start_id + block * 24 + replica * 6 + slice
```

so ids 0-5 are the six slices of block 0 with the first bandpass, 6-11 the same slices
with the second bandpass, and so on. Every non-seismic array is identical for the four
replicas of a slice. Next to the folders the generator writes `generator_config.json`
(settings of the latest run), `runs.jsonl`, `blocks.jsonl` (faults, unconformities,
bandpass centers and clip levels of every block) and `samples.csv` (one row per id).

## How a block is built

The steps follow the generator description; the details in *italics* were measured
on training_10_7_26, because the description refers to values that were not included
("the bandpass centers above", "the exact predicate above").

1. **Block.** Stratigraphy `tau = t0 + y + u(x)`, where `u` is a sum of Gaussian fold
   bumps, a regional tilt, fold growth with depth and lateral thickening. *The
   reference also stacks packages separated by erosional unconformities (with a tau
   gap); the generator does the same, with a probability you choose.* One to a few
   steep fault polylines (median dip 72 in the description, *75 measured with the same
   estimator on the reference*, which is the default; the Spec preset uses 72). The
   stroke is every pixel within the stroke radius of the trace; *1.55-1.85 px reproduces
   the 3-4 px wide strokes of 1.dat* (the description's "radius 2-3" gives 4-6 px; use
   the Spec preset or the Stroke radius setting for that). The throw is applied to the
   footwall beyond the stroke edge while the stroke keeps the hanging-wall values - the
   right edge for a fault dipping left. *In the reference 95% of faults are normal,
   throws are about 2-21 tau samples, they vary along the fault, and faults dip both
   ways (the right-dipping ones carry the throw on the left edge).* Faults are applied
   youngest first, so older faults are offset by younger ones. Optional drag bends
   reflectors into the fault.
2. **Six slices.** Each fault trace steps sideways between consecutive slices (about
   2 traces). *In the reference every fault has its own step, left or right, and the
   reflectors drift slightly between slices (a real 3D model).* "Reflector drift" = 0
   and "Step direction" = right reproduce the description literally.
3. **Four seismics per slice.** A blocky impedance log is defined along tau (layer
   thickness, marker beds, bright/dim intervals, lateral facies changes), turned into
   reflectivity and convolved down each trace with a wavelet whose center differs per
   replica. *Measured centers in the reference span about 0.06-0.20 cycles/sample
   (median 0.12) with a Ricker-like bandwidth; the default list is 0.075 / 0.105 /
   0.14 / 0.18 (19 / 26 / 35 / 45 Hz at 4 ms) with +/-12% jitter, shuffled between
   replicas.* *The reference does not simply standardise: every replica is
   rank-mapped to a standard normal (exact mean 0 / std 1, extremes near +/-4.7) and
   about 20% of replicas are then clipped, mostly at 2.1-2.8 - which is why their
   maxima fall "somewhere in +/-2.1 to +/-4.7".* Both behaviours are available
   (Normalization = gaussianize / standardize, clip probability, clip range and skew).
   Lateral gain, chaotic zones, fault-zone dimming and band-limited noise add realism.
4. **Dip.** `tau[y, x] - tau[y, x + S]`, lightly smoothed; columns without a partner
   repeat the last valid column.
5. **Straddle.** 1 where `x` and `x + S` are both off the stroke, `x + S` is inside the
   section, and a fault trace crosses the row strictly between them. *Rows past a fault
   tip (its rounded end cap) do not count; this reproduces the reference exactly
   except at those tip rows.*
6. **Scale weight.** 0 for `x >= 224 - S - 1`; 0 in a corridor around each stroke
   (2 px); 0 where the partner `x + S` falls in the stroke widened by 2 px left / 5 px
   right - with S = 12 the two corridors leave the one-pixel notch 3-4 columns before
   the stroke; *0 where the partner reflector leaves the section (top/bottom)*; and
   optionally 0 where `x` and `x + S` lie in different packages (the tau difference
   would contain the unconformity gap rather than a reflector shift).
7. **Halo.** Gaussian blur of the mask with radius S (sigma = S / 2), *divided by its
   99th percentile and clipped, so it peaks at 1*.
8. **Write** all arrays as little-endian float64 under the same id, repeating the
   non-seismic arrays across the four replicas.

Measured with `python -m synthetic_generator.cli compare <generated> <training_10_7_26>`
on 400 sampled ids each (default preset, seed 7):

| statistic | reference | generated |
|---|---|---|
| fault pixel fraction | 0.041 | 0.041 |
| fault angle (deg) | 74.9 | 75.6 |
| stroke width per row (px) | 3.63 | 3.69 |
| seismic std | 0.996 | 0.990 |
| seismic spectral centroid (cycles/sample) | 0.140 | 0.142 |
| straddle mean S=12 / 36 / 60 | 0.085 / 0.256 / 0.359 | 0.081 / 0.251 / 0.348 |
| weight mean S=12 / 36 / 60 | 0.757 / 0.642 / 0.546 | 0.735 / 0.630 / 0.550 |
| halo mean S=12 / 36 / 60 | 0.124 / 0.241 / 0.320 | 0.137 / 0.270 / 0.359 |

## SAM 3 and the fifteen arrays

SAM 3 takes a 3-channel image, but nothing has to be projected: only `Seis` is an
image. The Quick Fine-Tuner reads one `Seis` section, percentile-clips it to gray
(2-98%), feeds it as R = G = B and trains against `Fault` (binary mask, prompt
`fault`) or against layers cut from `TauAtT` (prompt `seismic facies`). The dip,
straddle, weight and halo arrays are supervision signals for a custom loss (dip
regression, weighting, fault proximity) and are ignored by the fine-tuner. The
**SAM 3 input** preview shows exactly this, plus the 2.5D option (previous / current /
next slice in R / G / B, as the app does for 3D volumes) and a 2 x 2 training mosaic.

## GUI

- **Settings tabs**: Dataset, Structure, Faults, Seismic, Realism, Labels. Hover a
  setting marked (i) for its help. Every range is drawn per block (or per fault /
  replica), so one dataset mixes many geological settings.
- **Presets**: Reference match (default), Spec (the literal 8-step description:
  fixed reflections, 2-trace step to the right, throw on the right edge, radius 2-3,
  standardise + clip 2.1-4.7), Clean and simple, Complex structure, Noisy field-like,
  Low / High frequency, Faults only. Load / Save keep settings as JSON (the same file
  the command line takes).
- **Preview views**: Overview (seismic, overlay, mask as written, tau, dip, label
  overlay), All arrays (the 15 files of one id), Slices (the six-slice sequence),
  Replicas and spectra (four bandpasses, amplitude spectra in cycles/sample and Hz,
  wiggle traces), Labels at one scale, SAM 3 input. Hover for values, double-click a
  tile for a large view.
- **Generate dataset** runs in a separate process with progress and ETA; **Stop**
  finishes the blocks in progress (a second click forces it). **Resume last run**
  reloads the folder's `generator_config.json` and writes only missing or incomplete
  blocks. With First id = -1 a new run appends after the existing ids. **Inspect**
  checks the folder (sizes, ids, replica grouping, Quick Fine-Tuner pairs) and prints
  statistics.

## Command line

```powershell
python -m synthetic_generator.cli dump-config --out my_settings.json
python -m synthetic_generator.cli generate --config my_settings.json --out D:\data\synthetic_75 --blocks 400
python -m synthetic_generator.cli preview --view sam3 --block 3 --out preview.png
python -m synthetic_generator.cli inspect D:\data\synthetic_75
python -m synthetic_generator.cli compare D:\data\synthetic_75 C:\...\Data\training_10_7_26
```

## Code

| Module | Role |
|---|---|
| `config.py` | every setting, presets, JSON, validation |
| `geology.py` | stratigraphy, unconformities, folds, faults, throw, slices |
| `seismic.py` | impedance, wavelets / bandpass, rendering, realism, normalization |
| `labels.py` | dip, straddle, scale weight, halo |
| `block.py` | one block = slices x replicas; folder names and id layout |
| `dataset.py` | parallel writer, resume, metadata |
| `checks.py` | dataset inspection and comparison |
| `preview.py` | preview images |
| `params.py`, `gui.py` | GUI controls and window |
| `cli.py` | command line |

Tests: `python -m pytest tests/test_synthetic_generator.py`.
