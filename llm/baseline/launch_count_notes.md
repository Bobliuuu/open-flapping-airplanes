# Launch counting notes (BASELINE v2)

FA's published figure: **514,562 kernel launches** for the whole run (4,000 training steps in 70.1 s + serving, Megatron + vLLM, each timed alone).
Their own arithmetic: 514,562 ~= 4,000 steps x ~116 + ~50k served tokens x ~1 (serving: 710 tok/s x 70 s ~= 49.8k tokens, ~1 CUDA-graph launch per token).
So at most ~128 launches per training step even if serving were free: the stack is highly fused / graphed. "Launch" is not defined in the video, hence three definitions.

## The three definitions (all measured with torch.profiler = CUPTI, chrome trace parsed by `launchcount.py`, 5 steady-state steps, per-unit = total/5)
- **(a) GPU kernel executions**: kernel activity records (trace category `kernel`). Kernels replayed inside a CUDA graph still count one each. memcpy/memset are not counted (reported separately in the JSON, ~20-40/step).
- **(b) host launch API calls**: `cudaLaunchKernel`, `cudaLaunchKernelExC`, `cudaLaunchCooperativeKernel`, `cudaGraphLaunch` plus driver-level `cuLaunchKernel/cuLaunchKernelEx/cuGraphLaunch` (TE and cuDNN launch through the driver API; they appear as separate, non-nested events, so no double counting: in eager mode (b) == (a) exactly). A graph launch is ONE call. Serving uses vLLM's default config (CUDA graphs on).
- **(c) host launch calls with the unit of work captured in ONE CUDA graph**: training: PyTorch eager = the *entire* step (batch gather, LR, fwd, bwd, fused AdamW) in one `torch.cuda.CUDAGraph`; Megatron-TE = fwd+bwd(+batch gather, grad-buffer zeroing) in one graph, Megatron's optimizer stays eager (its found-inf check synchronises the host, so it cannot be captured; `FullCudaGraphWrapper` in megatron-core also only wraps forward_backward, not the optimizer). Serving: vLLM decode with its full CUDA graph (vLLM's default mode FULL_AND_PIECEWISE already uses a full graph for batch-1 decode; `--cg full` gave identical counts).
Per-token serving counts are steady-state: (64-new-token generate minus 32-new-token generate)/32, so prefill is excluded.

## Measured per unit (A100, A100 kernel selection; cuDNN fused attention, TE LayerNormLinear/Linear/GELU, TE FusedAdam)
| system | model | (a) kernels/step | (b) host calls/step | (c) host calls/step, one graph |
|---|---|---|---|---|
| Megatron-core 0.16.1 + TE 2.20 | S (12L, 162M) | 589 | 589 | 99 (1 graph + 98 eager optimizer/misc) |
| Megatron-core 0.16.1 + TE 2.20 | F (4L, 408M) | 277 | 277 | 75 (1 graph + 74 eager optimizer/misc) |
| PyTorch eager (SDPA flash, fused AdamW) | S | 580 | 580 | 1 |
| PyTorch eager | F | 229 | 229 | 1 |
| (BASELINE v1, no TE) Megatron local spec | S | 958 | 958 | not measured |

| vLLM 0.31 decode, batch 1 | (a) kernels/token | (b) host calls/token (default, graphs on) | host calls/token, `enforce_eager` | (c) host calls/token (full graph) |
|---|---|---|---|---|
| S | 129 | 20 (1 graph + 11 cudaLaunchKernel + 8 cuLaunchKernelEx outside the graph) | 156 | 20 |
| F | 57 | 20 | 68 | 20 |
The ~19 launches/token outside the graph are vLLM's input preparation / sampling; a perfectly graphed decode would be 1/token (shown as "ideal" below).

## Whole-demo equivalents (4,000 steps + serving for the same wall time)
Served tokens: as specified, `serving tok/s x training seconds` (SM-restricted: training on 84 SMs, vLLM on 16 SMs, mean values). Because our baselines are slower than FA's
(233-353 s instead of 70 s) this inflates the served tokens to 117-192k vs FA's ~50k, and serving then dominates the counts. We therefore also give the same table with FA's own ~49.6k tokens (46.1 s x 1,076 tok/s). Megatron-TE rows are the headline system.

**With the spec's definition (tokens = tok/s x training seconds):**
| system / model | tokens | (a) kernels | (b) host calls | (c) one graph per step / token | (c) with ideal 1 launch/token | FA |
|---|---|---|---|---|---|---|
| Megatron-TE S | 120k | 17.89M | 4.77M | 2.81M | 0.52M | 514,562 |
| Megatron-TE F | 168k | 10.68M | 4.47M | 3.67M | 0.47M | 514,562 |
| PyTorch eager S | 117k | 17.39M | 4.66M | 2.35M | 0.12M | 514,562 |
| PyTorch eager F | 192k | 11.85M | 4.76M | 3.85M | 0.20M | 514,562 |

**With FA's served-token count fixed at 49,604 (like for like):**
| system / model | (a) kernels | (b) host calls | (c) one graph per step / token | (c) with ideal 1 launch/token | FA |
|---|---|---|---|---|---|
| Megatron-TE S | 8.76M (17x) | 3.35M (6.5x) | 1.39M (2.7x) | 0.45M (0.88x) | 514,562 |
| Megatron-TE F | 3.94M (7.7x) | 2.10M (4.1x) | 1.30M (2.5x) | 0.35M (0.68x) | 514,562 |
| PyTorch eager S | 8.72M | 3.32M | 1.00M | 0.05M | |
| PyTorch eager F | 3.75M | 1.91M | 1.00M | 0.05M | |
(training-only part: Megatron-TE S 2.36M/2.36M/0.40M, F 1.11M/1.11M/0.30M, PyTorch eager S 2.32M/2.32M/4k, F 0.92M/0.92M/4k for (a)/(b)/(c).)

## Which definition lands closest, and what it implies
- **(a) and (b) are far from 514,562 for every stack we can build** (4x to 35x too high, even the F model with TE). Even *training alone* is 0.9-2.4M, i.e. 2-5x FA's whole-demo number. FA's number therefore cannot be raw kernel executions or un-graphed host launches of a Megatron-style stack. For serving alone, vLLM executes 57-129 kernels per token (x ~50k tokens = 2.9-6.4M), which is already 5-12x FA's total.
- **(c) is the only definition that lands in range**: with the training step graphed and decode at ~1 launch/token (the ideal-graph column), a 4-layer model gives 0.35M (0.68x FA), the 12-layer 0.45M (0.88x). Megatron-TE with a graphed fwd+bwd leaves 75 (F) / 99 (S) launches per step, i.e. the same order as FA's implied ~116/step. A fully graphed step (1 launch/step) would be only 4,000 launches, so FA's ~116/step means *their* training step is partially graphed and/or fused to roughly 100 launches, but not one graph.
- Implication for their stack: ~100-130 launches per step is only achievable with (i) graph capture of most of the step (as in our Megatron-TE fwd+bwd graph, 75-99 residual launches, mostly the optimizer), or (ii) a very shallow model (our TE eager stack needs ~39 kernels per layer on top of ~120 fixed (embedding, head, loss, optimizer): F=4 layers gives 277, S=12 layers 589) together with heavy fusion. Both point to a shallow model (<= ~4-6 layers) with CUDA graphs, and serving at ~1 graph launch per token. Our model S at 12 layers cannot reach their count under (a) or (b); under (c) it can only reach it with near-perfect graphing of both training and decode.
- Kernel-count floor of our stacks per layer, from the traces: LN (TE fused with the following linear's input), 4 GEMMs fwd + 8 GEMMs bwd, cuDNN attention fwd/bwd, activation, residual adds, plus per-parameter elementwise kernels from Megatron's fp32 main-grad accumulation (1 add + 1 bf16 copy per parameter tensor per step, because apex's `gradient_accumulation_fusion` was not built) and TE FusedAdam multi-tensor kernels.

## Reproduce
`./run_v2.sh` (phases `count train_te train_pt_f train_graph serve`); counting alone: `MODEL=F python megatron_train.py --impl te --count3 --out x.json`, `MODEL=F python train_pytorch.py --count3 --out x.json`,
`python serve_vllm.py HF_DIR --count_launches --cg {default,eager,full} --out x.json`; `python aggregate_v2.py` writes `results/baseline_v2_a100_<date>.json` (`launch_counts.demo_equivalent` holds both token bases).
