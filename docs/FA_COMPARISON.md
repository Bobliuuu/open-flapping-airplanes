# open-flapping-airplanes vs Flapping Airplanes' demo: a fair comparison

Flapping Airplanes (FA) showed one demo video: training, serving and Doom in one kernel launch on an NVIDIA B200, against Megatron and
vLLM timed individually. Their numbers are read frame by frame from that video ([`benchmarks/fa_published.json`](../benchmarks/fa_published.json));
their code, model and settings are not public, and nobody outside FA has measured them. This page puts every open-flapping-airplanes measurement next to
theirs and marks what is like-for-like and what is not. open-flapping-airplanes is not affiliated with Flapping Airplanes.

## Bottom line

- **B200 vs B200 (this repo's demo, measured 2026-11-01 on Modal).** The demo reproduces: one launch on a B200 trains a model, serves text
  from its live weights and plays Doom at 35 fps, doing the work of 16,674 separate launches. The speedup does not. With each baseline
  timed alone on FA's SM share (their method: 116 SMs for training, 24 for serving, set here with CUDA MPS), the baseline is **22% faster** than open-flapping-airplanes
  at training and **17% faster** at serving, where FA report **34% slower** for both. On the whole GPU the baselines are 25% faster and
  4% slower.
- **That is not a fair test of their claim.** open-flapping-airplanes's demo trains a 115k-parameter character model, about 28,000 times less arithmetic per
  step than a 162M-parameter GPT, and its baseline is open-flapping-airplanes's own kernels launched one by one with no framework. FA's baseline is
  Megatron + vLLM running an LLM.
- **The closest like-for-like test is at LLM scale, but on an A100** (upstream runs, FA's method, open-flapping-airplanes's 162M GPT megakernel): by FA's
  metric the baseline is 37% slower for Megatron-core without Transformer Engine and 15% slower for vLLM, close to FA's 34%, but only
  1% slower for PyTorch eager. The size of the gain depends on the baseline.
- **Blackwell at LLM scale has one B300 run and no baseline:** 4,000 steps in 87.1 s against FA's 46.1 s, 1,762 against 1,076 tok/s served.

## 1. B200 vs B200

open-flapping-airplanes: `modal run benchmarks/run_modal.py --mode bench` on one NVIDIA B200 (148 SMs, driver 610.57.04, 1,965 MHz maximum SM clock),
5 repetitions x 6 s per measurement. Raw data [`b200_20261101.json`](../benchmarks/results/b200_20261101.json), derived table
[`summary_b200.md`](../benchmarks/results/summary_b200.md), container output [`b200_20261101_bench_stdout.txt`](../benchmarks/results/b200_20261101_bench_stdout.txt).
Baselines on FA's SM shares: `--mode mps`, the same per-op programs under CUDA MPS, 3 repetitions x 6 s, with 100% runs under the same
daemon as the control ([`b200_20261101_mps_baselines.json`](../benchmarks/results/b200_20261101_mps_baselines.json)).

| | Flapping Airplanes (B200, from video) | open-flapping-airplanes (B200, measured) | Like-for-like? |
|---|---|---|---|
| GPU | B200, 148 SMs | B200, 148 SMs | yes |
| In one launch | training + serving + Doom | training + serving + Doom | yes |
| Protocol | megakernel runs all three; each baseline timed alone | same | yes |
| Baseline | Megatron on 116 SMs, vLLM on 24 SMs | open-flapping-airplanes's task kernels launched per op (4 per step, 3 per token), on 116 and 24 SMs (MPS) | SM shares yes, framework **no** |
| Kernel launches for the same work | 1 vs 514,562 | 1 vs 16,674 (4,000 steps plus the tokens served meanwhile) | same definition, different workload |
| 4,000 training steps | 46.1 s vs 70.1 s: baseline **34% slower** | 0.225 s vs 0.184 s on 116 SMs: baseline **22% faster** (whole GPU: 0.180 s, 25% faster) | no (workload) |
| Serving | 1,076 vs 710 tok/s: baseline **34% slower** | 71,350 vs 83,127 tok/s on 24 SMs: baseline **17% faster** (whole GPU: 68,712 tok/s, 4% slower) | no (workload) |
| Time per training step | 11.5 ms | 56 µs | no |
| SMs per job (train / serve / Doom) | 116 / 24 / 8, static | 17.1 / 2.9 / 0.14 busy on average, dynamic | no |
| Doom | plays throughout | 35.0 fps throughout | yes |
| Model | LLM, about 100k vocabulary (loss 11.5 to 2.11), size unpublished | character MLP, 115,296 trainable parameters, 96-way softmax | **no** |

Also measured on the B200:

- Doom alone (`doomtest`, 120 frames) renders a frame in 5.0 ms (6.7 ms on the A100). Inside the megakernel and unpaced it reaches 254 fps.
- Run as two separate processes sharing the GPU (not FA's method, but what two frameworks on one GPU do), the baselines are 42% slower at
  training and 58% slower at serving; open-flapping-airplanes has 1.7x and 2.4x their throughput.
- A 20 s live run did 361,119 training steps (loss 1.38 to 0.13), generated 20,000 characters from the live weights and rendered 696 Doom
  frames: [`b200_20261101_live_state.json`](../benchmarks/results/b200_20261101_live_state.json), frames
  [0](../benchmarks/results/b200_20261101_live_frame0.png), [1](../benchmarks/results/b200_20261101_live_frame1.png),
  [2](../benchmarks/results/b200_20261101_live_frame2.png), [3](../benchmarks/results/b200_20261101_live_frame3.png).

## 2. What is still not like-for-like

| Difference | Effect | Can it be closed? |
|---|---|---|
| **Workload.** About 0.14 GFLOP per training step, against about 4 TFLOP for a 162M GPT at 4,096 tokens per step (FA's model size is not published). | A step lasts 56 µs, so the megakernel's software scheduling (claim, poll and fence for 144 tasks a step) is a large share of it. In FA's 11.5 ms steps the same overhead would be small. This works against the megakernel. | Only with an LLM workload (sections 3 and 4). |
| **Baseline.** Ours launches open-flapping-airplanes's own kernels with no framework; FA's is Megatron and vLLM (Python dispatch, hundreds of kernels a step). | Our baseline has none of the overhead a megakernel exists to remove. This also works against the megakernel. | Upstream has framework baselines on the A100 (section 3). |
| **SM shares.** FA pin each baseline to its share of the GPU. | Measured with CUDA MPS against 100% under the same daemon. Training on 116 SMs: unchanged (0.998x eager, 1.002x graphs). Serving on 24 SMs: **15-17% faster** (1.169x eager, 1.148x graphs); we have not investigated why. Section 1 uses these numbers. | Done for the baselines. open-flapping-airplanes's megakernel has no static-split mode; its jobs used about 17 and 3 SMs, and whether confining serving to 24 SMs would speed it up too is not measured. |
| **GPU load.** About 20 of 148 SMs busy, against FA's static split of all 148. | The run shows the mechanism, not a saturated GPU. | Only with an LLM workload. |
| **FA's numbers.** Read from a video; model, data and Megatron/vLLM settings unpublished. | Nothing can be matched exactly. | No. |

## 3. FA's ratio, from every open-flapping-airplanes run

FA's headline metric is how much slower each baseline, timed alone, is than the megakernel: 1 - baseline rate / megakernel rate.
Positive means the megakernel wins.

| Run | GPU | Model | Baseline | Training | Serving |
|---|---|---|---|---|---|
| Flapping Airplanes (video) | B200 | LLM, size unpublished | Megatron / vLLM on their SM shares | +34% | +34% |
| open-flapping-airplanes demo (this PR) | B200 | character MLP | open-flapping-airplanes's kernels, eager launches, on FA's SM shares (MPS) | -22.3% | -16.5% |
| open-flapping-airplanes demo (this PR) | B200 | character MLP | open-flapping-airplanes's kernels, CUDA graphs, on FA's SM shares (MPS) | -24.0% | -10.2% |
| open-flapping-airplanes demo (this PR) | B200 | character MLP | open-flapping-airplanes's kernels, eager launches, whole GPU | -25.0% | +3.7% |
| open-flapping-airplanes demo | A100 | character MLP | open-flapping-airplanes's kernels, eager launches, whole GPU | -13.1% | +2.1% |
| open-flapping-airplanes LLM (upstream) | A100 | 162M GPT | Megatron-core without TE on 84 SMs / vLLM on 16 SMs | +36.8% | +14.9% |
| open-flapping-airplanes LLM (upstream) | A100 | 162M GPT | PyTorch eager with flash attention, 84 SMs | +1.0% | - |
| open-flapping-airplanes demo (this PR), not FA's method | B200 | character MLP | both jobs as two processes sharing the GPU | +41.9% | +58.0% |

The gain depends on what the megakernel replaces. Against Megatron without Transformer Engine (unfused attention, about 960 kernels a
step), open-flapping-airplanes's LLM build gains about what FA report; against PyTorch eager with flash attention (about 580 kernels a step) it roughly ties;
against the same work launched as a few tight kernels (the toy demo) it loses. FA's +34% is consistent with a framework baseline, but their
baseline is unpublished, so this neither confirms nor refutes it.

Sources for the A100 LLM rows: [`llm/ofa/README.md`](../llm/ofa/README.md) (85 / 17 / 6 split, Doom on: 4,000 steps in 224 s, final
loss 2.149; decode 611 and 597 tok/s in two 600-step runs) and [`llm/baseline/README.md`](../llm/baseline/README.md) (each alone,
MPS-restricted: Megatron-core 354.6 ± 7.8 s, PyTorch eager 226.2 ± 11.0 s, vLLM 514 ± 2 tok/s; measured on a shared GPU, see its caveats).

## 4. Blackwell at LLM scale (a B300, not a B200)

Upstream ran the 162M GPT megakernel through Modal with `gpu="B200+"`, which placed it on a **B300** SXM6 (sm_103, 148 SMs):
[`llm/b200/results/mk_split_100a.txt`](../llm/b200/results/mk_split_100a.txt). FA's split of 116 / 24 / 8, Doom not linked in (its 8 SMs
idle), GEMMs on `mma.sync`; the repo has no validated run of the tcgen05 backend yet.

| | Flapping Airplanes (B200) | open-flapping-airplanes LLM (B300) |
|---|---|---|
| 4,000 training steps | 46.1 s (about 355k tok/s) | 87.1 s (188k tok/s): 1.89x longer, and longer than FA's Megatron baseline (70.1 s) |
| Serving during training | 1,076 tok/s | 1,762 tok/s (1.64x) |
| Final loss | 2.11 | 2.16 |
| Baseline on the same GPU | Megatron, vLLM | none yet |

Different chip, unknown FA model size and no baseline: this shows where open-flapping-airplanes's LLM build stands, not a like-for-like result.

## 5. What a strict B200-vs-B200 at LLM scale needs

1. The LLM megakernel ([`llm/b200/`](../llm/b200/README.md)) on an actual B200 (`gpu="B200"`, not `"B200+"`), with Doom linked in.
2. Megatron with Transformer Engine, and vLLM, on the same B200, limited to 116 / 24 SMs ([`llm/b200/baseline/run_all.sh`](../llm/b200/baseline/run_all.sh)).
   CUDA MPS works on Modal: on the B200, 79% gives exactly 116 SMs and 17% gives 24 (this PR's `--mode mps` run); upstream's B300 probe
   also got 116 at 79% ([`llm/b200/results/probe.json`](../llm/b200/results/probe.json)).
3. A validated tcgen05 GEMM backend, so the megakernel uses Blackwell's tensor cores the way cuBLAS does.

Upstream's runbook estimates 6 to 8 GPU-hours for that session.

## Reproduce the B200 runs

```bash
uv tool install modal && modal setup                 # once; needs a Modal account
modal run benchmarks/run_modal.py                    # smoke test: Doom alone, then all three jobs for 4 s
modal run benchmarks/run_modal.py --mode bench       # 5 x 6 s -> benchmarks/results/b200_<date>.json (about 7 B200-minutes)
modal run benchmarks/run_modal.py --mode mps         # baselines limited to 116 / 24 SMs with CUDA MPS
modal run benchmarks/run_modal.py --mode live        # 20 s live demo: Doom frames, generated text, loss
python3 benchmarks/compare.py benchmarks/results/b200_20261101.json --summary benchmarks/results/summary_b200.md --no-site
```
