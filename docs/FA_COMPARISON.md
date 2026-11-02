# open-flapping-airplanes vs Flapping Airplanes' demo: a fair comparison

Flapping Airplanes (FA) showed one demo video: an LLM trained, served from its live weights and Doom played, all in one kernel launch on an
NVIDIA B200, against Megatron and vLLM timed individually. Their numbers are read frame by frame from that video
([`benchmarks/fa_published.json`](../benchmarks/fa_published.json)); their code, model and settings are not public, and nobody outside FA has
measured them. This page compares open-flapping-airplanes with them as closely as we can: the same GPU, the same SM split, the same kind of model and the same
baseline frameworks, all measured by us. open-flapping-airplanes is not affiliated with Flapping Airplanes.

## Bottom line

- **Like-for-like on one B200 at LLM scale** ([`b200-run/`](../b200-run/README.md), measured 2026-11-02): open-flapping-airplanes's megakernel trains a
  162M-parameter GPT for 4,000 steps in **95.8 s** while it serves and runs Doom on a 116 / 24 / 8 SM split. Megatron-core with Transformer
  Engine, alone on the same 116 SMs, takes **62.4 s**. By FA's metric the baseline is **54% faster** at training, where FA report theirs
  **34% slower**.
- **Serving goes the other way.** open-flapping-airplanes generates **1,764 tok/s** on its 24 SMs while training; vLLM alone on 24 SMs makes **684 tok/s**,
  **61% slower** (FA: 34% slower).
- **Our baselines match theirs.** Our Megatron (62.4 s) and vLLM (684 tok/s) are within 11% and 4% of FA's (70.1 s, 710 tok/s), so the
  setups are probably comparable; FA have not published theirs.
- **The training gap is open-flapping-airplanes's GEMMs.** They use `mma.sync`, which reaches 13-33% of cuBLAS on the B200. The repo's Blackwell (`tcgen05`)
  GEMM backend passed its first B200 test here, but inside the megakernel it runs one block per SM and the step gets slower. FA's
  megakernel (46.1 s) is faster than every baseline we measured.
- **The toy demo** (a 115k-parameter character model, section 3) does not reproduce FA's gain either.

## 1. LLM scale, B200 vs B200

| | Flapping Airplanes (from their video) | open-flapping-airplanes (measured) | Like-for-like? |
|---|---|---|---|
| GPU | B200, 148 SMs | B200, 148 SMs (Modal) | yes |
| Model | LLM, about 100k-token vocabulary (loss 11.5 to 2.11), size unpublished | GPT, 162.4M parameters, cl100k vocabulary (loss 11.71 to 2.16) | close; FA's size unknown |
| Training | 4,000 steps, about 4,096 tokens per step | 4,000 steps x 4,096 tokens, AdamW, TinyStories | yes; FA's data unknown |
| One launch runs | training + serving + Doom on 116 / 24 / 8 SMs | the same (`--split b200`) | yes |
| Baselines | Megatron on 116 SMs and vLLM on 24 SMs, each alone | Megatron-core 0.16.1 + Transformer Engine 2.20 on 116 SMs and vLLM 0.33 on 24 SMs, each alone (CUDA MPS) | yes; FA's versions and settings unknown |
| Data, initial weights, batch order | unknown | identical on both sides | - |

| | Flapping Airplanes | open-flapping-airplanes |
|---|---|---|
| 4,000 training steps, megakernel | 46.1 s | 95.8 s (3 runs: 95.78, 95.76, 95.81) |
| 4,000 training steps, Megatron on 116 SMs | 70.1 s | 62.4 s (3 runs, sd 0.2) |
| **Training: how much slower the baseline is** | **+34%** | **-54%** (Megatron is faster) |
| Serving, megakernel | 1,076 tok/s | 1,764 tok/s |
| Serving, vLLM on 24 SMs | 710 tok/s | 684 tok/s |
| **Serving: how much slower the baseline is** | **+34%** | **+61%** |
| Doom | plays throughout | 35.0 fps throughout |
| Final loss | 2.11 | megakernel 2.158, Megatron 2.154 (mean of 3) |
| Kernel launches for the same work | 1 vs 514,562 | 1 vs 1.2M to 8.4M (three definitions) |

Other baselines on the same B200, each alone for 4,000 steps: PyTorch eager with flash attention, 52.5 s on 116 SMs and 45.4 s on all 148;
Megatron + Transformer Engine, 57.3 s on all 148; Megatron-core without Transformer Engine, 92.9 s on 116. open-flapping-airplanes training alone on all 148
SMs: 79.5 s. Per-run numbers, launch counts and the GEMM measurements: [`b200-run/README.md`](../b200-run/README.md).

## 2. What is still not like-for-like

| Difference | Effect |
|---|---|
| FA's model size, data and framework settings are unpublished | Their Megatron and vLLM are within 11% and 4% of ours, so the setups are probably close; this cannot be checked |
| Kernel quality: open-flapping-airplanes's GEMMs reach 13-33% of cuBLAS on the B200 | Not a protocol difference, but it decides the training result. The `tcgen05` backend is correct and faster for some GEMMs, yet only fits one block per SM in the megakernel ([`b200-run/README.md`](../b200-run/README.md)) |
| SM shares set with CUDA MPS (79% gives 116 SMs, 17% gives 24) | FA did not say how they limited theirs |
| One B200 on Modal | FA's clocks and power limit are unknown |

## 3. The toy demo on the B200

The original demo ([`a100-demo/`](../a100-demo/README.md)) trains a 115k-parameter character model, about 0.14 GFLOP per step (28,000 times
less than the GPT above), and compares against its own kernels launched one by one ([`benchmarks/run_modal.py`](../benchmarks/run_modal.py),
[`summary_b200.md`](../benchmarks/results/summary_b200.md), [`b200_20261101_mps_baselines.json`](../benchmarks/results/b200_20261101_mps_baselines.json)):

| Baseline, timed by FA's metric | Training | Serving |
|---|---|---|
| open-flapping-airplanes's kernels, eager launches, alone on FA's SM shares (MPS) | -22.3% | -16.5% |
| the same with CUDA graphs | -24.0% | -10.2% |
| open-flapping-airplanes's kernels, eager launches, alone on the whole GPU | -25.0% | +3.7% |
| both jobs as two processes sharing the GPU (not FA's method) | +41.9% | +58.0% |

A step of this model lasts 56 µs, so the megakernel's scheduling (claim, poll and fence for 144 tasks a step) costs more than four kernel
launches do. One launch still runs all three programs, with Doom at 35 fps (5.0 ms per frame alone). A 20 s live run:
[`b200_20261101_live_state.json`](../benchmarks/results/b200_20261101_live_state.json), frames
[0](../benchmarks/results/b200_20261101_live_frame0.png), [1](../benchmarks/results/b200_20261101_live_frame1.png),
[2](../benchmarks/results/b200_20261101_live_frame2.png), [3](../benchmarks/results/b200_20261101_live_frame3.png).

## 4. FA's ratio, from every open-flapping-airplanes run

FA's headline metric is how much slower each baseline, timed alone, is than the megakernel: 1 - baseline rate / megakernel rate. Positive
means the megakernel wins.

| Run | GPU | Model | Baseline | Training | Serving |
|---|---|---|---|---|---|
| Flapping Airplanes (video) | B200 | LLM, size unpublished | Megatron / vLLM on their SM shares | +34% | +34% |
| **open-flapping-airplanes LLM ([`b200-run`](../b200-run/README.md))** | **B200** | **162M GPT** | **Megatron-core + TE on 116 SMs / vLLM on 24 SMs** | **-53.6%** | **+61.2%** |
| open-flapping-airplanes LLM (`b200-run`) | B200 | 162M GPT | PyTorch eager with flash attention on 116 SMs | -82.5% | - |
| open-flapping-airplanes LLM (`b200-run`) | B200 | 162M GPT | Megatron-core without TE on 116 SMs | -3.1% | - |
| open-flapping-airplanes LLM (upstream) | A100 | 162M GPT | Megatron-core without TE on 84 SMs / vLLM on 16 SMs | +36.8% | +14.9% |
| open-flapping-airplanes LLM (upstream) | A100 | 162M GPT | PyTorch eager with flash attention on 84 SMs | +1.0% | - |
| open-flapping-airplanes demo | B200 | character MLP | open-flapping-airplanes's kernels, eager, on FA's SM shares | -22.3% | -16.5% |
| open-flapping-airplanes demo | A100 | character MLP | open-flapping-airplanes's kernels, eager, whole GPU | -13.1% | +2.1% |

At training, the result follows the baseline's kernels. Against Megatron without Transformer Engine (unfused attention, about 960 kernels a
step) the megakernel wins on the A100 and is about even on the B200; against Megatron with Transformer Engine and PyTorch with flash attention,
whose GEMMs run at cuBLAS speed on the B200, it loses. At LLM-scale serving, where batch-1 decoding is bound by launches and latency, it wins
on both GPUs, by more than FA report. FA's training gain needs a megakernel whose GEMMs keep up with cuBLAS; open-flapping-airplanes's do not yet.

Sources for the A100 LLM rows: [`llm/ofa/README.md`](../llm/ofa/README.md) (85 / 17 / 6 split, Doom on: 4,000 steps in 224 s, final
loss 2.149; decode 611 and 597 tok/s in two 600-step runs) and [`llm/baseline/README.md`](../llm/baseline/README.md) (each alone,
MPS-restricted: Megatron-core 354.6 ± 7.8 s, PyTorch eager 226.2 ± 11.0 s, vLLM 514 ± 2 tok/s; measured on a shared GPU, see its caveats).

## 5. A B300 run (upstream)

Upstream ran the same megakernel through Modal with `gpu="B200+"`, which placed it on a B300 SXM6 (sm_103, 148 SMs): 87.1 s for 4,000 steps
on the 116 / 24 / 8 split with Doom compiled out, and 1,762 tok/s served ([`llm/b200/results/mk_split_100a.txt`](../llm/b200/results/mk_split_100a.txt)).
The B200 run gives the same loss (2.1575 at step 3,999) and the same serving rate, and takes 10% longer.

## Reproduce

```bash
modal run b200-run/modal_app.py --step <prep|env|mk|train|serve|gemm|report>    # the LLM-scale comparison (b200-run/README.md)
modal run benchmarks/run_modal.py --mode <smoke|bench|mps|live>                   # the toy demo (benchmarks/README.md)
```
