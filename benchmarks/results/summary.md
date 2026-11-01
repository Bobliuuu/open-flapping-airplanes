# Benchmark summary (NVIDIA A100-SXM4-40GB, 5 reps x 6 s, 2026-11-01T04:19:19Z)

| metric | mean | sd |
|---|---|---|
| megakernel, all three at once: train_steps_s | 13507.6 | 214.1 |
| megakernel, all three at once: train_tok_s | 3457953.5 | 54809.5 |
| megakernel, all three at once: infer_tok_s | 53211.6 | 751.2 |
| megakernel, all three at once: doom_fps | 35.0 | 0.0 |

| baseline (each job alone unless stated) | train steps/s | infer tok/s | baseline slower than megakernel (train / infer) | 4,000 steps take | GPU-equivalents needed |
|---|---|---|---|---|---|
| per-op kernels, eager launches (the method the video uses) | 15282 | 52070 | -13.1% / 2.1% | 0.262 s | 1.91 |
| per-op kernels, CUDA graphs (best case for the baseline) | 16296 | 60343 | -20.6% / -13.4% | 0.245 s | 1.71 |
| both jobs in one process, two CUDA-graph streams | 14960 | 53377 | -10.8% / -0.3% | 0.267 s | n/a |
| two separate processes sharing the GPU (time-sliced) | 7465 | 27278 | 44.7% / 48.7% | 0.536 s | n/a |

Megakernel: 4,000 steps take 0.296 s. Flapping Airplanes (B200): 46.1 s vs 70.1 s, baseline 34% slower, both jobs 34% slower, 1 vs 514,562 launches.
