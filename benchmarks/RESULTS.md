# Interactive performance baseline

Measured on 2026-08-19 with the representative `SEGY0000.sgy` volume
(145 inlines x 145 crosslines x 1951 samples), PyTorch 2.5.1 CUDA 12.1,
Transformers 5.15.0, and an NVIDIA GeForce RTX 3080 Laptop GPU.

- Active-slice image preparation: 0.87 seconds, paid before the first click.
- Warm point prompt: 14.5 ms p50 and 14.9 ms p95 across five runs.
- Warm full Streamlit script rerun: 71 ms in Streamlit AppTest.
- Estimated warm server-side click path: under 100 ms before browser transport.
- Point tracker peak allocated VRAM: 3.04 GB.
- Full 145-frame bfloat16 propagation: 96.6 seconds cold (1.50 FPS);
  a repeated run completed in 105.4 seconds.
- Frame construction took 0.06 seconds and video-session setup took 0.65
  seconds, so retaining mutable sessions would add risk without material gain.
- Fifteen-frame bfloat16 sample: 8.25 seconds cold and 9.85 seconds warm.
- The same fifteen-frame float32 sample: 26.0 seconds cold and 33.2 seconds
  warm. Bfloat16 was 3.15x faster cold and 3.38x faster warm, with the same
  aggregate mask coverage (93.74%).

The warm-click result is below the one-second React migration threshold.
Propagation remains dominated by sequential tracker inference, but CUDA
bfloat16 exceeds the planned 2x improvement target and the UI now reports
incremental progress instead of showing only a blocking spinner.

`torch.compile` was also evaluated, but PyTorch Inductor cannot compile this
model in the current Windows environment because Triton is unavailable. It is
therefore left as an explicit benchmark option rather than enabled in the app.
