# Large seismic file support (any SEG-Y / NPY too big for RAM)

ampSmall.sgy (15.2 GB) is the reference case, but the solution must work for any large file.

## Investigation
- [x] Measure the reference file: ampSmall.sgy is 1701 IL x 2101 XL x 1001 samples, IBM float, IL/XL in header bytes 17/25 (189/193 are zero)
- [x] Find why the screen is black: the fast listing always reports SEG-Y files as 2D, so any large 3D volume shows up as a giant line the viewer cannot display
- [x] Find the memory blowup: `load_any` plus `normalize_to_uint8` peak at about 5x the float32 size, so any file larger than about RAM/5 fails
- [x] Find the propagation limits: RGB frames, processor tensors and masks all grow with the full axis length
- [x] Check the hardware: 64 GB RAM, 28 threads, RTX 3500 Ada 12 GB, 407 GB free disk

## Decision
- [x] Present the solution options and get the user's choice: hybrid cache, three axis-ordered memmaps, windowed propagation
- [x] Generalize the plan to any large file (any header layout and sort order, irregular grids, large 2D lines, large NPY, memory-based routing, auto-sized windows)

- [x] Acknowledge the SAM3 requirement: no changes to model or inference code, byte-identical inputs for small files, additive API only, cache builds never touch the GPU

## Implementation
- [x] Record the baseline: run the existing test suite and the API smoke test before any change (one stale test was failing already; it is fixed)
- [x] Generic geometry detection: expanded header-byte candidates plus an override file, inline- or crossline-major sorting, missing-trace grids, fast sampled listing
- [x] Source adapters: `SegySource` (any sample format) and `NpySource` (mmap) behind one streaming interface
- [x] `seismic_app/volume_cache.py`: sampled percentile, streamed 3-order uint8 memmaps, trace-to-cell map, manifest, disk-space check, build queue (an interrupted build is detected and redone; about 45 s for ampSmall)
- [x] `SeismicVolume` abstraction (in-memory, direct and cached; 3D and paged 2D) wired into `server/main.py`, with a memory-based large-file threshold
- [x] Background build queue at startup (the open file goes first), plus `/api/file-status`; `/api/meta` no longer blocks
- [x] Windowed propagate, refine, resweep and export, with the window auto-sized from the RAM budget
- [x] Detailed logging: routing, geometry, cache, build progress (MB/s, ETA, RAM, disk), slice timings, propagation memory
- [x] Frontend: progress overlay, axis gating, 2D page navigation, slice spinner/errors, propagation window control
- [x] Synthetic tests across layouts and formats, large 2D and large NPY; run the existing suite (75 passed)
- [x] SAM3 regression gate: full test suite, API smoke test, and click / propagate / refine / auto-segment on SEGY0000.sgy (slice images identical to the old path)
- [x] Verify end to end on ampSmall.sgy, SEGY0000.sgy, L4.sgy and a synthetic crossline-major file
