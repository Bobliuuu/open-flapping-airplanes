# Benchmark summary (NVIDIA B200, 5 reps x 6 s, 2026-11-01T17:13:35Z)

| metric | mean | sd |
|---|---|---|
| megakernel, all three at once: train_steps_s | 17788.6 | 332.7 |
| megakernel, all three at once: train_tok_s | 4553879.2 | 85178.4 |
| megakernel, all three at once: infer_tok_s | 71349.9 | 382.4 |
| megakernel, all three at once: doom_fps | 35.0 | 0.0 |

| baseline (each job alone unless stated) | train steps/s | infer tok/s | baseline slower than megakernel (train / infer) | 4,000 steps take | GPU-equivalents needed |
|---|---|---|---|---|---|
| per-op kernels, eager launches (the method the video uses) | 22239 | 68712 | -25.0% / 3.7% | 0.180 s | 1.84 |
| per-op kernels, CUDA graphs (best case for the baseline) | 22111 | 66873 | -24.3% / 6.3% | 0.181 s | 1.87 |
| both jobs in one process, two CUDA-graph streams | 21450 | 64109 | -20.6% / 10.1% | 0.186 s | n/a |
| two separate processes sharing the GPU (time-sliced) | 10328 | 29961 | 41.9% / 58.0% | 0.387 s | n/a |

Megakernel: 4,000 steps take 0.225 s. Flapping Airplanes (B200): 46.1 s vs 70.1 s, baseline 34% slower, both jobs 34% slower, 1 vs 514,562 launches.
