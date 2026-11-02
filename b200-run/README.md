# b200-run: Triad against Megatron and vLLM on one NVIDIA B200 (LLM scale, Flapping Airplanes' protocol)

Measured on 2026-11-02 on Modal with `gpu="B200"` (NVIDIA B200, 148 SMs, driver 610.57.04, 1,965 MHz maximum SM clock). Everything runs
from [`modal_app.py`](modal_app.py); raw outputs are in [`results/`](results/). The comparison with Flapping Airplanes (FA) is in
[`../docs/FA_COMPARISON.md`](../docs/FA_COMPARISON.md). Not affiliated with Flapping Airplanes.

## What is matched

| | Megakernel (Triad) | Baselines |
|---|---|---|
| GPU | one B200 | one B200, same type |
| Model | 162.4M-parameter GPT "S" (12 layers, d=768, cl100k vocabulary padded to 100,352), [`../llm/SPEC.md`](../llm/SPEC.md) | the same model in Megatron-core, PyTorch and vLLM |
| Training | 4,000 steps x 4,096 tokens, AdamW, TinyStories in cl100k tokens | the same data, initial weights, batch order and optimizer schedule |
| How it runs | ONE launch: training on 116 SMs, serving from the live weights on 24, Doom on 8 (`--split b200`) | each job ALONE, limited with CUDA MPS to the same share: 116 SMs for training, 24 for serving |
| Code | [`../llm/b200/megakernel`](../llm/b200/megakernel/README.md): `tcgen05` GEMMs (`triad_tc2`, `triad_tc3`); the first runs used `mma.sync` (`triad_s`) | [`../llm/b200/baseline`](../llm/b200/baseline/run_all.sh) scripts, unchanged |

MPS at 79% and 17% gives exactly 116 and 24 SMs on this GPU ([`../benchmarks/results/b200_20261101_mps_baselines.json`](../benchmarks/results/b200_20261101_mps_baselines.json)).
Every baseline run started on an idle GPU, with only its own process (and the MPS server) on it (`results/runs_v2/*.meta.json`).

## Results

Training, 4,000 steps:

| System | SMs | 4,000 steps | tok/s | Final loss |
|---|---|---|---|---|
| **Triad megakernel, `tcgen05` GEMMs, 3 stages** (`triad_tc3`), while serving and running Doom | 116 of 116 / 24 / 8 | **48.1 s** (47.95, 48.17, 48.16) | 341k | 2.1575 (all 3 runs) |
| **Triad megakernel, `tcgen05` GEMMs, 2 stages** (`triad_tc2`), while serving and running Doom | 116 of 116 / 24 / 8 | **52.4 s** (52.39, 52.39, 52.39) | 313k | 2.1575 (all 3 runs) |
| Triad megakernel, `mma.sync` GEMMs (`triad_s`, the first runs), while serving and running Doom | 116 of 116 / 24 / 8 | 95.8 s (95.78, 95.76, 95.81) | 171k | 2.1575 (all 3 runs) |
| **Megatron-core 0.16.1 + Transformer Engine 2.20, CUDA graphs** (forward + backward in one graph, optimizer eager), alone | 116 | **43.1 s** (43.62, 42.90, 42.85) | 380k | 2.146 (2.142-2.149) |
| **Megatron-core 0.16.1 + Transformer Engine 2.20**, eager, alone | 116 | **62.4 s** (sd 0.2, 3 runs) | 263k | 2.154 (2.146-2.158) |
| PyTorch with flash attention, whole step in one CUDA graph, alone | 116 | 50.9 s (51.40, 50.65, 50.70) | 322k | 2.149 (2.145-2.152) |
| PyTorch eager with flash attention, alone | 116 | 52.5 s (sd 0.02, 3 runs) | 312k | 2.152 (2.144-2.157) |
| Megatron-core without Transformer Engine, alone | 116 | 92.9 s (1 run) | 176k | 2.139 |
| Megatron-core + Transformer Engine, alone | 148 | 57.3 s | 286k | 2.144 |
| PyTorch eager, alone | 148 | 45.4 s | 361k | 2.148 |
| Triad megakernel, training only: `triad_tc3` / `triad_tc2` / `triad_s` | 148 | 40.0 s / 42.6 s / 79.5 s | 410k / 385k / 206k | 2.1575 |

Megatron-core depends on its container. In a second container, with 16 CPU cores reserved, it took 129.2 s on 116 SMs (129.0, 130.3,
128.3), 128.4 s on 148, and 127.6 s without Transformer Engine, while PyTorch eager in that same container stayed at 52.8 s and 45.8 s
([`results/runs_v2_cpu16/`](results/runs_v2_cpu16/)); the configuration and attention backend are identical in both logs. The comparison
uses Megatron's faster times, which favours the baseline. Each megakernel build took the same time in every container it ran in (`triad_s`
95.7 to 96.0 s). The CUDA-graph baselines (`--step train_graphs`, [`results/runs_v3_graph/`](results/runs_v3_graph/)) ran in both kinds of
container and agree within 0.6% ([`results/runs_v3_graph_cpu16/`](results/runs_v3_graph_cpu16/): 43.4 s and 50.9 s); with graphs, Megatron
no longer depends on its container.

Serving, batch 1, FA's prompt "What is a megakernel?", 256-token answers. The megakernel asks the prompt again from its live weights after
every answer (`--answer-len 256`; [`results/megakernel_tc2_fa/`](results/megakernel_tc2_fa/), [`results/megakernel_tc3_fa/`](results/megakernel_tc3_fa/),
[`results/megakernel_fa/`](results/megakernel_fa/) for `triad_s`); its rate counts sampled tokens only, as vLLM's does:

| System | SMs | tok/s |
|---|---|---|
| **Triad megakernel, `triad_tc2`**, during training, 3 runs | 24 | **1,741** (1,738, 1,744, 1,742) |
| **Triad megakernel, `triad_tc3`**, during training, 3 runs | 24 | **1,660** (1,665, 1,655, 1,661) |
| Triad megakernel, `triad_s`, during training, 3 runs | 24 | 1,783 (1,780, 1,787, 1,783) |
| **vLLM 0.33.0**, CUDA graphs on, alone, 16 reserved CPU cores, 3 runs | 24 | **1,537** (1,571, 1,510, 1,529) |
| vLLM 0.33.0, alone, 16 reserved CPU cores | 148 | 1,403 (1,314, 1,492) |

With "Once upon a time" and continuous decoding, in the same 16-core container: megakernel 1,764 tok/s, vLLM 1,438 on 24 SMs (1,503,
1,521, 1,290). vLLM at batch 1 is bound by its CPU: our first vLLM runs, in a container without reserved cores, made 684 tok/s
([`results/runs_v2/`](results/runs_v2/)), and an earlier version of this page compared against that. It is kept, not compared.

Serving slows as training speeds up: the 24 serving SMs share L2 and HBM with a trainer that now moves the same bytes in half the time,
and `triad_tc3`'s third pipeline stage takes shared memory that the serving SMs would use as L1 (`TRIAD_CARVEOUT=100` on `triad_s`, which
leaves 28 KB of L1, served 1,590 tok/s instead of 1,783 in a 600-step run).

Doom ran at 35.0 fps throughout every megakernel run. Every build gives the same loss, bit for bit, at every printed step (the `tcgen05`
GEMMs round exactly as the `mma.sync` ones do), and for steps 0-19 it is within 6e-5 of the A100 reference (step 0: 11.710616 against
11.710649; step 19: 8.821558 against 8.821497).

Against FA's published B200 numbers (FA's metric: how much slower the baseline is than the megakernel, 1 - baseline rate / megakernel rate):

| | FA | Triad, `triad_tc3` | Triad, `triad_tc2` | Triad, `triad_s` (first runs) |
|---|---|---|---|---|
| Training, Megatron on 116 SMs, eager | 70.1 s vs 46.1 s: **+34%** | 62.4 s vs 48.1 s: **+23%** | 62.4 s vs 52.4 s: **+16%** | 62.4 s vs 95.8 s: -54% |
| Training, Megatron on 116 SMs, CUDA graphs | not reported | 43.1 s vs 48.1 s: **-11.5%** | 43.1 s vs 52.4 s: **-21.5%** | 43.1 s vs 95.8 s: -122% |
| Serving, vLLM on 24 SMs | 710 vs 1,076 tok/s: **+34%** | 1,537 vs 1,660 tok/s: **+7%** | 1,537 vs 1,741 tok/s: **+12%** | 1,537 vs 1,783 tok/s: +14% |

`triad_tc3` trains within 4% of FA's 46.1 s. Our eager Megatron lands at 0.89x of FA's 70.1 s; FA do not say whether theirs used CUDA graphs.
With them, Megatron + Transformer Engine is the fastest training system we measured (43.1 s), ahead of both Triad builds. Our vLLM, given enough CPU, is 2.2x FA's 710 tok/s; FA's figure is close to what ours made
without reserved cores. Launches for the work of one megakernel run (4,000 steps and the about 91,000 tokens `triad_tc2` served): 585 host launch calls per eager
Megatron step (97 with CUDA graphs) and 20.1 per vLLM token (141 kernels), so 2.2M to 4.2M host launch calls and 15.2M kernels in total
(`launch_counts` in [`results/baseline_v2_b200_20261102.json`](results/baseline_v2_b200_20261102.json); its `demo_equivalent` totals used
the CPU-starved vLLM rate), against 1.

What the model answers. The megakernel's answers to "What is a megakernel?" go from random tokens at step 0 to children's stories
([`results/megakernel_fa/answers_split_r1.txt`](results/megakernel_fa/answers_split_r1.txt)); vLLM's copy of the trained model writes
" It is a big, round thing. ..." ([`results/runs_v2/vllm_S_text.json`](results/runs_v2/vllm_S_text.json)). The model only ever saw
TinyStories. FA's model, from its answers in their video, was trained on text that defines things, megakernels included, and ends up
answering the question; this is not reproduced here. Nor is their greedy decoding (their decode loop shows `temperature=0.0`; ours samples
at 0.8).

## The GEMMs: from `mma.sync` to `tcgen05`

GEMMs are most of a training step. The first runs used `mma.sync`, which reaches 13-33% of cuBLAS on the B200; the `tcgen05` backend
([`../llm/b200/gemm_tcgen05`](../llm/b200/gemm_tcgen05/README.md)) passed its first test but made the megakernel slower. Two things held it back:

1. **One block per SM.** The occupancy API returns 1 block per SM for any kernel that contains a `tcgen05` instruction, whatever its
   registers and shared memory, and the megakernel sized its grid from it. The hardware runs two such blocks per SM; what serializes them
   is the TMEM allocation permit: while one block holds it, the other block's `tcgen05.alloc` waits until the holder exits. Each block now
   allocates its TMEM columns once, gives up the permit at once (`tcgen05.relinquish_alloc_permit`) and keeps the columns for the whole
   run; the grid is sized from registers and shared memory.
2. **The epilogue.** Each thread stored its own row of the 128 x 128 tile, so every warp store touched 32 rows. The accumulator now goes
   through a swizzled fp32 staging tile in the idle pipeline buffers and leaves one tile row per warp store.

Then: the barriers moved to static shared memory (2 stages fit the same 132 KB carveout as `mma.sync`), and the embedding backward,
whose most common tokens made one task do hundreds of serial row updates, accumulates each row in registers (1.55 to 0.72 ms per step).

GEMM test on this B200 ([`results/gemm_v2/`](results/gemm_v2/), 15 shapes of the model, all PASS, persistent == grid bit-exact):

| GEMM backend | Speed (share of cuBLAS) |
|---|---|
| `mma.sync` | 13-33% |
| `tcgen05`, first version ([`results/gemm/`](results/gemm/)) | 15-42% |
| `tcgen05` now | **34-73%** (forward and data-gradient GEMMs 0.48-0.73, weight gradients 0.34-0.55) |

The megakernel during development, 600-step runs on the split with FA's prompt, one change at a time (the full runs above are the
evidence for the last row; the intermediate builds' logs are not kept here):

| Build | Blocks per SM | ms per step | Serving |
|---|---|---|---|
| `mma.sync` (`triad_s`) | 2 | 23.9 | 1,783 tok/s |
| `tcgen05`, first version | 1 | 25.9 | 975 tok/s |
| + TMEM once per block, 2 blocks per SM | 2 | 19.6 | 1,477 tok/s |
| + staged epilogue, 3 stages / 2 stages | 2 | 12.3 / 13.3 | 1,514 / 1,753 tok/s |
| + static barriers (3 stages fit a smaller carveout), per-row embedding backward = `triad_tc3` / `triad_tc2` | 2 | 11.9 / 13.1 | 1,679 / 1,742 tok/s |

The full runs above measure the last row. `triad_tc2` keeps more L1 for the serving SMs; `triad_tc3` trains 8% faster. The earlier 600-step
runs of the first `tcgen05` version are in `results/megakernel_triad_*_600/`.

## Reproduce

```bash
uv tool install modal && modal setup                     # once
modal run b200-run/modal_app.py --step prep              # CPU: data, initial weights, batch order, vocab -> Volume triad-b200-data
modal run b200-run/modal_app.py --step env               # check torch / Transformer Engine / Megatron-core / vLLM on a B200
modal run b200-run/modal_app.py --step mk                # megakernel: 20-step check, 3 x 4,000 steps on the split, training only
modal run b200-run/modal_app.py --step mk --serving fa --tag megakernel_fa --no-trainonly   # serving FA's prompt, 256-token answers
modal run b200-run/modal_app.py --step train             # Megatron-core and PyTorch, alone, 116 SMs and whole GPU (--sub for another folder)
modal run b200-run/modal_app.py --step train_graphs      # the same with CUDA graphs, in a default and a 16-core container -> results/runs_v3_graph*/
modal run b200-run/modal_app.py --step serve             # vLLM, alone, 24 SMs and whole GPU, launch counts, text sample
modal run b200-run/modal_app.py --step serve_both        # vLLM, both prompts in one container -> results/runs_v2_cpu16/
modal run b200-run/modal_app.py --step gemm --tag gemm_v2   # the GEMM test (mma.sync control, then tcgen05) -> results/gemm_v2/
modal run b200-run/modal_app.py --step tccmp             # triad_tc2 and triad_tc3: loss check, 3 x 4,000 steps with FA's prompt, training only
modal run b200-run/modal_app.py --step video             # full-run dashboard recording (FA's prompt) -> video/out/ (not committed)
modal run b200-run/modal_app.py --step report            # aggregate_v2.py + fa_compare.py, copy results/ here
```
Images: megakernel on CUDA 12.8.1 with clang 14 (Doom), builds `triad_s`, `triad_tc2` and `triad_tc3`; training on NGC PyTorch 26.10 (torch 2.15, Transformer Engine 2.20.2) plus
megatron-core 0.16.1; serving on CUDA 13.0 with vllm 0.33.0 (torch 2.14). The baseline steps (`train`, `serve`, `serve_both`) now
reserve 16 CPU cores (`CPUS` in `modal_app.py`); the first `train` and `serve` runs (`results/runs_v2/`) did not. All the runs above took
roughly 105 B200-minutes (the two `tcgen05` builds' full runs and the GEMM test about 15 of them).

Notes: `results/megakernel/loss20.txt` is the console output of the 20-step check; its per-step loss file was overwritten by a name clash (fixed in
`modal_app.py`), and steps 0-19 are in `results/megakernel/loss_split_r1.txt`. `*.log` files are not committed (repo `.gitignore`); the JSON files
carry the numbers.
