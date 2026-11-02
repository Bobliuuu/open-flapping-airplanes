# open-flapping-airplanes vs Flapping Airplanes' demo on a B200

Flapping Airplanes (FA) showed one demo video: an LLM trained, served from its live weights and Doom played, all in one kernel launch on an
NVIDIA B200, against Megatron and vLLM timed individually. Their numbers are read frame by frame from that video
([`benchmarks/fa_published.json`](../benchmarks/fa_published.json)); their code, model and settings are not public, and nobody outside FA has
measured them. This page compares open-flapping-airplanes with them as closely as we can: the same GPU, the same SM split, the same kind of model and the same
baseline frameworks, all measured by us. open-flapping-airplanes is not affiliated with Flapping Airplanes.

## Bottom line

- **Like-for-like on one B200 at LLM scale** ([`b200-run/`](../b200-run/README.md), measured 2026-11-02): open-flapping-airplanes's megakernel trains a
  162M-parameter GPT for 4,000 steps in **48.1 s** (`tcgen05` GEMMs, 3 pipeline stages) or **52.4 s** (2 stages) while it serves and runs
  Doom on a 116 / 24 / 8 SM split. Megatron-core with Transformer Engine, alone on the same 116 SMs, takes **62.4 s** as frameworks run by
  default (eager): by FA's metric that baseline is **23% / 16% slower**, where FA report theirs **34% slower**. **With CUDA graphs the same
  Megatron takes 43.1 s and is 11.5% / 21.5% faster than open-flapping-airplanes** (and faster than FA's reported 46.1 s); PyTorch with a whole-step CUDA graph
  takes 50.9 s (open-flapping-airplanes's 3-stage build is 5.5% faster, the 2-stage build 2.9% slower). FA do not say how their Megatron ran; their 70.1 s is
  close to our eager time. The first build, with `mma.sync` GEMMs, took 95.8 s.
- **Serving: open-flapping-airplanes wins by less than FA report.** Answering FA's prompt from its live weights, open-flapping-airplanes makes **1,660 / 1,741 tok/s** (3 / 2
  stages) on its 24 SMs while it trains; vLLM alone on 24 SMs, with 16 CPU cores reserved, makes **1,537 tok/s**, **7% / 12% slower** (FA:
  34% slower). The `mma.sync` build served 1,783 tok/s; the faster trainer leaves the serving SMs less memory bandwidth and, with 3 stages,
  less L1. An earlier version of this page compared against a vLLM run on a CPU-starved container (684 tok/s) and reported 61%.
- **Our eager Megatron matches theirs; our vLLM is faster.** Our eager Megatron (62.4 s) is within 11% of FA's (70.1 s). Our vLLM, given enough CPU,
  makes 2.2x FA's 710 tok/s; FA's number is close to what ours made without reserved cores. FA have not published their setup.
- **The training result is the GEMMs.** `mma.sync` reaches 13-33% of cuBLAS on the B200; the Blackwell (`tcgen05`) backend now reaches
  34-73%. It first ran at 1 block per SM because the occupancy API reports 1 for any kernel with `tcgen05` instructions and a block that
  keeps its TMEM allocation permit stalls the next one; allocating TMEM once per block and relinquishing the permit gives 2 blocks per SM,
  and a staged, coalesced epilogue did the rest ([`b200-run/README.md`](../b200-run/README.md#the-gemms-from-mmasync-to-tcgen05)). open-flapping-airplanes's
  3-stage build is 4% behind FA's reported 46.1 s; the fastest training run we measured is Megatron + TE with CUDA graphs, 43.1 s.
- **The toy demo** (a 115k-parameter character model, section 3) does not reproduce FA's gain either.

## 1. LLM scale, B200 vs B200

| | Flapping Airplanes (from their video) | open-flapping-airplanes (measured) | Like-for-like? |
|---|---|---|---|
| GPU | B200, 148 SMs | B200, 148 SMs (Modal) | yes |
| Model | LLM, about 100k-token vocabulary (loss 11.5 to 2.11), size unpublished | GPT, 162.4M parameters, cl100k vocabulary (loss 11.71 to 2.16) | close; FA's size unknown |
| Training | 4,000 steps, about 4,096 tokens per step | 4,000 steps x 4,096 tokens, AdamW, TinyStories | yes; FA's data unknown |
| One launch runs | training + serving + Doom on 116 / 24 / 8 SMs | the same (`--split b200`) | yes |
| Baselines | Megatron on 116 SMs and vLLM on 24 SMs, each alone | Megatron-core 0.16.1 + Transformer Engine 2.20 on 116 SMs and vLLM 0.33 on 24 SMs, each alone (CUDA MPS) | yes; FA's versions and settings unknown |
| Serving | "What is a megakernel?", answered again and again from the live weights, greedy | the same prompt, asked again after every 256-token answer (`--answer-len 256`), sampled at temperature 0.8 | yes, except greedy decoding |
| GEMMs | unpublished | `tcgen05` (Blackwell tensor cores, TMEM), 34-73% of cuBLAS | - |
| Training data | text that defines things: their model answers the question correctly by step ~3,300 | TinyStories: our model writes children's stories | **no** |
| Model | SwiGLU MLP (their code, shown in the video) | GELU MLP | no |
| Data, initial weights, batch order | unknown | identical on both sides | - |

| | Flapping Airplanes | open-flapping-airplanes, `tcgen05` 3 stages | open-flapping-airplanes, `tcgen05` 2 stages | open-flapping-airplanes, `mma.sync` (first build) |
|---|---|---|---|---|
| 4,000 training steps, megakernel | 46.1 s | 48.1 s (47.95, 48.17, 48.16) | 52.4 s (52.39, 52.39, 52.39) | 95.8 s (95.78, 95.76, 95.81) |
| 4,000 training steps, Megatron on 116 SMs, eager | 70.1 s (settings unpublished) | 62.4 s (3 runs, sd 0.2) | 62.4 s | 62.4 s |
| **Training: how much slower the eager baseline is** | **+34%** | **+23%** | **+16%** | -54% (Megatron is faster) |
| 4,000 training steps, Megatron on 116 SMs, CUDA graphs | not reported | 43.1 s (43.62, 42.90, 42.85) | 43.1 s | 43.1 s |
| **Training: how much slower the CUDA-graph baseline is** | - | **-11.5%** (Megatron is faster) | **-21.5%** (Megatron is faster) | -122% |
| Serving FA's prompt, megakernel | 1,076 tok/s | 1,660 tok/s (1,665, 1,655, 1,661) | 1,741 tok/s (1,738, 1,744, 1,742) | 1,783 tok/s (1,780, 1,787, 1,783) |
| Serving FA's prompt, vLLM on 24 SMs | 710 tok/s | 1,537 tok/s (3 runs, 16 CPU cores reserved) | 1,537 tok/s | 1,537 tok/s |
| **Serving: how much slower the baseline is** | **+34%** | **+7%** | **+12%** | +14% |
| Doom | plays throughout | 35.0 fps throughout | 35.0 fps throughout | 35.0 fps throughout |
| Final loss | 2.11 | megakernel 2.158 (every build, bit for bit), Megatron 2.154 (mean of 3) | | |
| Kernel launches for the same work (4,000 steps + the tokens open-flapping-airplanes served; setup not counted) | 1 vs 514,562 | 1 vs 2.0M (Megatron with CUDA graphs) to 3.9M (eager) host launch calls; 13.6M kernels | 1 vs 2.2M to 4.2M; 15.2M kernels | |

Other baselines on the same B200, each alone for 4,000 steps: PyTorch eager with flash attention, 52.5 s on 116 SMs and 45.4 s on all 148;
Megatron + Transformer Engine, 57.3 s on all 148; Megatron-core without Transformer Engine, 92.9 s on 116. open-flapping-airplanes training alone on all 148
SMs: 40.0 s (`tcgen05`, 3 stages), 42.6 s (2 stages), 79.5 s (`mma.sync`). Per-run numbers, launch counts and the GEMM measurements: [`b200-run/README.md`](../b200-run/README.md).
Counted with Nsight Systems, a whole `ofa_tc2` process makes 3 kernel launches: two one-off setup kernels and the megakernel, which runs
all of the training, serving and Doom ([`b200-run/results/launches_ofa_tc2/`](../b200-run/results/launches_ofa_tc2/nsys_launches.json)).

## 2. What is still not like-for-like

| Difference | Effect |
|---|---|
| FA's model size, data and framework settings are unpublished | Their Megatron is within 11% of ours, so the setups are probably close; this cannot be checked |
| Training data: FA's model learns to answer "What is a megakernel?"; ours trains on TinyStories | Different loss curve (FA: 5.03 at step 445; ours: about 3.7) and different served text; the final loss is close (2.11 vs 2.16) |
| Host CPU: the framework baselines are CPU-bound | Megatron took 62.4 s in one container and 129 s in another; vLLM made 684 tok/s without reserved cores and about 1,500 with 16. We compare against the faster numbers |
| Kernel quality: open-flapping-airplanes's GEMMs reach 34-73% of cuBLAS on the B200 (`tcgen05`; `mma.sync`: 13-33%) | Not a protocol difference, but it decides the training result ([`b200-run/README.md`](../b200-run/README.md)) |
| Serving vs training on the same chip | open-flapping-airplanes's serving rate falls as its trainer gets faster (shared L2 and HBM; with 3 stages, less L1): 1,783 tok/s with the `mma.sync` trainer, 1,741 / 1,660 with the `tcgen05` ones. Each baseline runs alone, with the whole memory system to itself |
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
| **open-flapping-airplanes LLM ([`b200-run`](../b200-run/README.md)), `tcgen05` 3 stages** | **B200** | **162M GPT** | **Megatron-core + TE (eager) on 116 SMs / vLLM on 24 SMs** | **+22.9%** | **+7.4%** |
| **open-flapping-airplanes LLM (`b200-run`), `tcgen05` 2 stages** | **B200** | **162M GPT** | **Megatron-core + TE (eager) on 116 SMs / vLLM on 24 SMs** | **+16.0%** | **+11.7%** |
| **open-flapping-airplanes LLM (`b200-run`), `tcgen05` 3 / 2 stages** | **B200** | **162M GPT** | **Megatron-core + TE with CUDA graphs on 116 SMs** | **-11.5% / -21.5%** | - |
| open-flapping-airplanes LLM (`b200-run`), `tcgen05` 3 / 2 stages | B200 | 162M GPT | PyTorch with flash attention, whole step in one CUDA graph, on 116 SMs | +5.5% / -2.9% | - |
| open-flapping-airplanes LLM (`b200-run`), `tcgen05` 3 / 2 stages | B200 | 162M GPT | PyTorch eager with flash attention on 116 SMs | +8.4% / +0.2% | - |
| open-flapping-airplanes LLM (`b200-run`), `tcgen05` 3 / 2 stages | B200 | 162M GPT | Megatron-core without TE on 116 SMs | +48.2% / +43.6% | - |
| open-flapping-airplanes LLM (`b200-run`), `mma.sync` (first build) | B200 | 162M GPT | Megatron-core + TE on 116 SMs / vLLM on 24 SMs | -53.6% | +13.8% |
| open-flapping-airplanes LLM (upstream) | A100 | 162M GPT | Megatron-core without TE on 84 SMs / vLLM on 16 SMs | +36.8% | +14.9% |
| open-flapping-airplanes LLM (upstream) | A100 | 162M GPT | PyTorch eager with flash attention on 84 SMs | +1.0% | - |
| open-flapping-airplanes demo | B200 | character MLP | open-flapping-airplanes's kernels, eager, on FA's SM shares | -22.3% | -16.5% |
| open-flapping-airplanes demo | A100 | character MLP | open-flapping-airplanes's kernels, eager, whole GPU | -13.1% | +2.1% |

At training, the result follows the GEMMs and how the baseline launches its kernels. With `mma.sync` GEMMs (13-33% of cuBLAS) the
megakernel beat only Megatron without Transformer Engine. With `tcgen05` GEMMs (34-73% of cuBLAS) it beats the frameworks as they run by
default (eager), Megatron + Transformer Engine by 16-23% against FA's 34%; but Megatron + Transformer Engine with CUDA graphs, which
removes most of its launch overhead, is 12-22% faster than open-flapping-airplanes. At LLM-scale serving, where batch-1 decoding is bound by launches and
latency and vLLM already uses CUDA graphs, it wins on both GPUs, by 7-15%, less than the 34% FA report; on the B200 the faster trainer
costs the serving SMs some bandwidth.

Sources for the A100 LLM rows: [`llm/ofa/README.md`](../llm/ofa/README.md) (85 / 17 / 6 split, Doom on: 4,000 steps in 224 s, final
loss 2.149; decode 611 and 597 tok/s in two 600-step runs) and [`llm/baseline/README.md`](../llm/baseline/README.md) (each alone,
MPS-restricted: Megatron-core 354.6 ± 7.8 s, PyTorch eager 226.2 ± 11.0 s, vLLM 514 ± 2 tok/s; measured on a shared GPU, see its caveats).

## 5. A B300 run (upstream)

Upstream ran the same megakernel through Modal with `gpu="B200+"`, which placed it on a B300 SXM6 (sm_103, 148 SMs): 87.1 s for 4,000 steps
on the 116 / 24 / 8 split with Doom compiled out, and 1,762 tok/s served ([`llm/b200/results/mk_split_100a.txt`](../llm/b200/results/mk_split_100a.txt)).
The B200 run gives the same loss (2.1575 at step 3,999) and the same serving rate, and takes 10% longer.

## Reproduce

```bash
modal run b200-run/modal_app.py --step <prep|env|mk|tccmp|train|serve|gemm|report>    # the LLM-scale comparison (b200-run/README.md)
modal run benchmarks/run_modal.py --mode <smoke|bench|mps|live>                   # the toy demo (benchmarks/README.md)
```
