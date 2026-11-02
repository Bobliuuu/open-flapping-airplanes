# b200-run: open-flapping-airplanes against Megatron and vLLM on one NVIDIA B200 (LLM scale, Flapping Airplanes' protocol)

Measured on 2026-11-02 on Modal with `gpu="B200"` (NVIDIA B200, 148 SMs, driver 610.57.04, 1,965 MHz maximum SM clock). Everything runs
from [`modal_app.py`](modal_app.py); raw outputs are in [`results/`](results/). The comparison with Flapping Airplanes (FA) is in
[`../docs/FA_COMPARISON.md`](../docs/FA_COMPARISON.md). Not affiliated with Flapping Airplanes.

## What is matched

| | Megakernel (open-flapping-airplanes) | Baselines |
|---|---|---|
| GPU | one B200 | one B200, same type |
| Model | 162.4M-parameter GPT "S" (12 layers, d=768, cl100k vocabulary padded to 100,352), [`../llm/SPEC.md`](../llm/SPEC.md) | the same model in Megatron-core, PyTorch and vLLM |
| Training | 4,000 steps x 4,096 tokens, AdamW, TinyStories in cl100k tokens | the same data, initial weights, batch order and optimizer schedule |
| How it runs | ONE launch: training on 116 SMs, serving from the live weights on 24, Doom on 8 (`--split b200`) | each job ALONE, limited with CUDA MPS to the same share: 116 SMs for training, 24 for serving |
| Code | [`../llm/b200/megakernel`](../llm/b200/megakernel/README.md), `mma.sync` GEMMs | [`../llm/b200/baseline`](../llm/b200/baseline/run_all.sh) scripts, unchanged |

MPS at 79% and 17% gives exactly 116 and 24 SMs on this GPU ([`../benchmarks/results/b200_20261101_mps_baselines.json`](../benchmarks/results/b200_20261101_mps_baselines.json)).
Every baseline run started on an idle GPU, with only its own process (and the MPS server) on it (`results/runs_v2/*.meta.json`).

## Results

Training, 4,000 steps:

| System | SMs | 4,000 steps | tok/s | Final loss |
|---|---|---|---|---|
| **open-flapping-airplanes megakernel**, while serving and running Doom | 116 of 116 / 24 / 8 | **95.8 s** (95.78, 95.76, 95.81) | 171k | 2.1575 (all 3 runs) |
| **Megatron-core 0.16.1 + Transformer Engine 2.20**, alone | 116 | **62.4 s** (sd 0.2, 3 runs) | 263k | 2.154 (2.146-2.158) |
| PyTorch eager with flash attention, alone | 116 | 52.5 s (sd 0.02, 3 runs) | 312k | 2.152 (2.144-2.157) |
| Megatron-core without Transformer Engine, alone | 116 | 92.9 s (1 run) | 176k | 2.139 |
| Megatron-core + Transformer Engine, alone | 148 | 57.3 s | 286k | 2.144 |
| PyTorch eager, alone | 148 | 45.4 s | 361k | 2.148 |
| open-flapping-airplanes megakernel, training only | 148 | 79.5 s | 206k | 2.1575 |

Serving, batch 1:

| System | SMs | tok/s |
|---|---|---|
| **open-flapping-airplanes megakernel**, during the training run above | 24 | **1,764** (1,762, 1,767, 1,762) |
| **vLLM 0.33.0**, CUDA graphs on, alone | 24 | **684** (685, 683, 683) |
| vLLM 0.33.0, alone | 148 | 682 |

Doom ran at 35.0 fps throughout every megakernel run (3,344 frames). The megakernel's loss is the same in every run and, for steps 0-19,
within 6e-5 of the A100 reference (step 0: 11.710616 against 11.710649; step 19: 8.821558 against 8.821497).

Against FA's published B200 numbers (FA's metric: how much slower the baseline is than the megakernel, 1 - baseline rate / megakernel rate):

| | FA | open-flapping-airplanes |
|---|---|---|
| Training, Megatron on 116 SMs | 70.1 s vs 46.1 s: **+34%** | 62.4 s vs 95.8 s: **-54%** (Megatron is faster) |
| Serving, vLLM on 24 SMs | 710 vs 1,076 tok/s: **+34%** | 684 vs 1,764 tok/s: **+61%** |

Our Megatron and vLLM land at 0.89x and 0.96x of FA's, so the baselines are comparable. Launches for the work of one megakernel run:
585 kernels per Megatron step and 141 per vLLM token, 1.2M to 8.4M in total depending on the definition (`launch_counts` in
[`results/baseline_v2_b200_20261102.json`](results/baseline_v2_b200_20261102.json)), against 1.

Text from the trained model in vLLM, prompted with FA's "What is a megakernel?" (TinyStories, 4,000 steps): greedy " It is a big, round
thing. It is a big, round thing. ..."; sampled " A nominated is a big duck. It is charming for food. ..." ([`results/runs_v2/vllm_S_text.json`](results/runs_v2/vllm_S_text.json)).

## Why the megakernel loses at training: its GEMMs

The GEMM test ([`results/gemm/`](results/gemm/)) times the model's 15 GEMM shapes against cuBLAS on this B200:

| GEMM backend | Correct | Speed |
|---|---|---|
| `mma.sync` (what the megakernel uses) | PASS, all 15 shapes | 13-33% of cuBLAS |
| `tcgen05` ([`../llm/b200/gemm_tcgen05`](../llm/b200/gemm_tcgen05/README.md), its first run on Blackwell) | PASS, all 15 shapes, every layout | 15-42% of cuBLAS: weight gradients 1.4-1.9x faster than `mma.sync`, several forward GEMMs slower |

Built into the megakernel, the `tcgen05` backend runs only 1 block per SM instead of 2 (the occupancy query says 1, and a 2-stage pipeline
that halves its shared memory did not change that), and the step gets slower (600 steps on the 116 / 24 / 8 split, `results/megakernel_ofa_*_600/`):

| Build | Blocks per SM | ms per step | Serving |
|---|---|---|---|
| `mma.sync` | 2 | 23.9 | 1,762 tok/s |
| `tcgen05`, weight gradients only | 1 | 31.0 | 998 tok/s |
| `tcgen05`, all GEMMs | 1 | 25.9 | 975 tok/s |
| `tcgen05`, all GEMMs, 2 stages | 1 | 26.0 | 1,039 tok/s |

So the comparison above uses the `mma.sync` build, open-flapping-airplanes's fastest on this GPU. FA's megakernel (46.1 s) is faster than every baseline we
measured, which points at Blackwell-native kernels on their side.

## Reproduce

```bash
uv tool install modal && modal setup                     # once
modal run b200-run/modal_app.py --step prep              # CPU: data, initial weights, batch order, vocab -> Volume ofa-b200-data
modal run b200-run/modal_app.py --step env               # check torch / Transformer Engine / Megatron-core / vLLM on a B200
modal run b200-run/modal_app.py --step mk                # megakernel: 20-step check, 3 x 4,000 steps on the split, training only
modal run b200-run/modal_app.py --step train             # Megatron-core and PyTorch, alone, 116 SMs and whole GPU
modal run b200-run/modal_app.py --step serve             # vLLM, alone, 24 SMs and whole GPU, launch counts, text sample
modal run b200-run/modal_app.py --step gemm              # the GEMM test; --step tccmp compares the tcgen05 megakernel builds
modal run b200-run/modal_app.py --step report            # aggregate_v2.py + fa_compare.py, copy results/ here
```
Images: megakernel on CUDA 12.8.1 with clang 14 (Doom); training on NGC PyTorch 26.10 (torch 2.15, Transformer Engine 2.20.2) plus
megatron-core 0.16.1; serving on CUDA 13.0 with vllm 0.33.0 (torch 2.14). All the runs above took roughly 45 B200-minutes.

Notes: `results/megakernel/loss20.txt` is the console output of the 20-step check; its per-step loss file was overwritten by a name clash (fixed in
`modal_app.py`), and steps 0-19 are in `results/megakernel/loss_split_r1.txt`. `*.log` files are not committed (repo `.gitignore`); the JSON files
carry the numbers.
