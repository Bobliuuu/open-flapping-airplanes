# open-flapping-airplanes LLM baseline: Megatron + vLLM equivalents on the A100 (FA "KVM DOOM" method)

Reproduces the *baseline* side of the Flapping Airplanes demo (training with Megatron, serving with vLLM, each timed individually on an
otherwise idle GPU, no Doom) for model "S" (see `../SPEC.md`: 12L, d=768, 12H, T=512, V=100352, tied head, 162.4M params) on this
A100-40GB (108 SMs). Results: `results/baseline_a100_<date>.json` (schema mirrors `benchmarks/fa_published.json`, mean/sd/n per metric),
raw per-run files in `results/runs/`. Golden reference for the megakernel: `golden/README.md`.

## Rerun everything
```
# once: venv + data (CPU/network)
python3 -m venv ~/venv-baseline && ~/venv-baseline/bin/pip install tiktoken transformers vllm megatron-core
#   TinyStories-valid.txt and the first 300 MB (HTTP range 0-314572799) of TinyStories-train.txt -> ~/data/
~/venv-baseline/bin/python prep_data.py          # -> ~/data/tinystories_cl100k_{train,valid}.u32 (uint32, <|endoftext|>=100257 between stories)
./run_baseline.sh                                 # A100 defaults TOTAL_SMS=108 TRAIN_SMS=85 INFER_SMS=17
TOTAL_SMS=148 TRAIN_SMS=116 INFER_SMS=24 CUDA_ARCH=sm_100 ./run_baseline.sh   # B200 (FA split)
```
`run_baseline.sh` env: `REPS` (3), `STEPS` (4000), `PHASES` (`golden train train_mps serve serve_mps text`), `IDLE_WAIT_S` (600).
Each timed run waits for `nvidia-smi` util < 5% (up to IDLE_WAIT_S, then runs anyway and is flagged `noisy_not_idle_before_start`), and a sampler
records the number of compute processes on the GPU during the run (`max_compute_procs_seen`; 1 = only our process, under MPS the MPS server may
add 1). SM restriction uses a user-level CUDA MPS daemon (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = round(100*SMS/TOTAL_SMS)`; started and stopped by the script,
pipe/log dirs `~/mps_baseline`). `baseline_check_sms` (check_sms.cu) verifies how many SMs a grid really lands on: MPS rounds, 79% gave 84 SMs (target 85), 16% gave 16 (target 17).

## Files
- `common.py` model S (eager PyTorch), init, LR schedule, deterministic batch sampler, optimizer; SM counts from env.
- `prep_data.py` tokenization; `train_pytorch.py` eager baseline (+ `--golden`); `megatron_train.py` Megatron-core GPTModel (local spec);
  `make_hf.py` HF GPT2 checkpoint builder; `serve_vllm.py` vLLM benchmark/text; `aggregate.py` -> results JSON.

## Method details
- Data: cl100k_base, 73,910,428 train tokens (344,421 stories, first 300 MB of TinyStories-train, trailing partial story dropped), 4,549,031 valid tokens (21,990 stories). Batch s = 8 sequences, starts `np.random.default_rng(1234).integers(0, ntok-513, size=(4000,8))[s]`; x = tok[st:st+512], y = tok[st+1:st+513]. Same sampler in both trainers (tokens preloaded on GPU, gathered on GPU).
- Optimizer: AdamW lr 6e-4, betas (0.9,0.95), eps 1e-8, wd 0.1 on all >=2-D tensors (including wte, wpe), none on gammas, no grad clipping, 100-step linear warmup then cosine to 6e-5 at step 4000. bf16 autocast, fp32 master weights.
- Timing: wall clock around the 4000-step loop with a single sync at start and end (no per-step host sync; losses stored on GPU), excludes process start-up/data load.
- PyTorch eager uses `F.scaled_dot_product_attention` (flash). Megatron-core 0.16.1 runs WITHOUT Transformer Engine/apex (local layer spec, unfused masked softmax attention `masked_softmax_fusion=False`, bf16 params + fp32 main params in Megatron's own optimizer and DDP wrapper, world size 1). Megatron has LayerNorm *bias* (zero-initialised, trained; +19,200 params) unlike SPEC; initial weights are copied from the golden init (QKV re-interleaved per head), initial loss matches PyTorch to 1e-4.
- Kernel launches: kernel activity records from CUPTI via `torch.profiler` over 3 steady-state steps x 4000 (training); for vLLM over a 64-token generate (default config, CUDA graphs on, so `host cudaGraphLaunch` is ~1/token while ~130 kernels execute per token inside it). vLLM runs in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), batch 1, 256 greedy new tokens from a 4-token prompt, prompt/encoding via tiktoken (`skip_tokenizer_init`).
- vLLM serving weights: step-4000 weights of the full-GPU PyTorch run (converted by `make_hf.py`; HF GPT2 Conv1D layout == SPEC layout), `kv_cache_memory_bytes=2GiB` (vLLM's free-memory profiling asserts on a shared GPU).
- Venv torch is 2.13+cu130 (pulled by vLLM), not the system 2.7.

## Results (A100-40GB, 2026-11-01, n=3, mean +- sd; details in results/baseline_a100_20261101.json)
| | full GPU (108 SMs) | SM-restricted (MPS; train 84 SMs actual, infer 16 actual) |
|---|---|---|
| Megatron-core train, 4000 steps | 317.7 +- 9.4 s, 51.6k +- 1.5k tok/s | 354.6 +- 7.8 s, 46.2k +- 1.0k tok/s |
| PyTorch eager train, 4000 steps | 207.9 +- 0.4 s, 78.8k +- 0.1k tok/s | 226.2 +- 11.0 s, 72.5k +- 3.5k tok/s |
| vLLM decode, batch 1, 256 tok | 535 +- 8 tok/s | 514 +- 2 tok/s |

Final loss (step 4000, single step): Megatron 2.146/2.152, PyTorch 2.143/2.146 (FA: 2.1094). Kernel launches: PyTorch eager 581 kernels/step (2.32M per 4000 steps),
Megatron local-spec 958/step (3.83M); vLLM 129 kernels executed per token (22 host launch calls/token with CUDA graphs).
Caveat: the GPU is shared with other agents; some runs saw 2-3 compute processes on the GPU (see `max_compute_procs_seen_per_run`), which includes our own
process/MPS server, so the figures are probably but not provably clean. Megatron without TE is much slower than eager PyTorch here (unfused attention), so it is a weaker baseline than FA's (which presumably used TE).
vLLM decode is CPU/launch-bound at this model size, hence almost insensitive to SM count.

---
# BASELINE v2 (2026-11-01): Transformer-Engine Megatron, config F, launch counting

Everything above (v1) is unchanged (`MODEL` defaults to S; `results/baseline_a100_20261101.json` untouched). v2 adds: Megatron-core 0.16.1 **with Transformer Engine 2.20.2** (`megatron_train.py --impl te`),
a second model **F** (L=4, d=2048, H=32, MLP 8192, T=512, V=100352, tied, 407,914,496 params, `MODEL=F`), CUDA-graph variants, three launch-count definitions.
Results: `results/baseline_v2_a100_20261101.json` (mean/sd/n, raw runs in `results/runs_v2/`), launch arithmetic: `launch_count_notes.md`, F golden: `golden/README.md` (`golden/F/`).

## What was installed into ~/venv-baseline (and how)
The venv's torch is 2.13+cu130, the system nvcc is 12.8, so builds use the venv's pip CUDA-13 toolchain (`nvidia-cuda-nvcc` 13.4) through a symlink farm `~/cuda13_home` (bin/include/lib64 -> `site-packages/nvidia/cu13`; created outside the venv).
- `transformer-engine 2.20.2` + `transformer-engine-cu13 2.20.2` (prebuilt wheels, `--no-deps`) and the PyTorch extension **built from the sdist**: `transformer_engine_torch` (`CUDA_HOME=~/cuda13_home NVTE_FRAMEWORK=pytorch MAX_JOBS=16 pip install --no-build-isolation --no-deps transformer_engine_torch-2.20.2.tar.gz`, ~40 min, CPU only).
- `flash-attn 2.8.3.post2` built from source for sm_80 only (`FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80 NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`; the define is needed because the pip nvcc is 13.4 but the pip CUDA headers are 13.0), ~3 h. It was installed at 18:27, i.e. **after** all timed TE runs but one (which had already imported TE), so all headline TE numbers use TE's cuDNN FusedAttention (`DotProductAttention: Selected backend = FusedAttention (sub-backend 1)`); flash-attn runs were done afterwards with `--attn flash` (below).
- apex: NOT installed. Not needed: Megatron's optimizer uses `transformer_engine.pytorch.optimizers.FusedAdam`. Consequence: `gradient_accumulation_fusion=False` (needs apex' `fused_weight_gradient_mlp_cuda` for the tied output layer), so fp32 main-grad accumulation costs one extra elementwise add per parameter tensor per step.
- Nothing else installed. `onnx`/`onnxscript`, which TE 2.20 imports unconditionally, are missing; `te_env.py` (imported first by `megatron_train.py`) puts inert stand-in modules into `sys.modules` (ONNX export is unused) and preloads the venv's `libcublas.so.13` etc. Torch/vLLM are untouched.

## Mapping SPEC onto TE modules
`get_gpt_layer_with_transformer_engine_spec(use_te_activation_func=True)`: `TELayerNormColumnParallelLinear` (LN fused into the QKV linear, and into fc1), `TERowParallelLinear` (proj, fc2), `TEDotProductAttention` (causal, no dropout), activation = TE GELU op, `cross_entropy_loss_fusion=True` (`te` impl), `masked_softmax_fusion`, bf16 params + fp32 main params, TE FusedAdam (AdamW mode, wd 0.1, betas (0.9,0.95), eps 1e-8, no clipping), learned absolute positions, tied head via Megatron's embedding/output sharing.
- **LN gamma only**: TE's LayerNorm always owns a beta (`layer_norm_bias`). Beta is zero-initialised and **frozen (`requires_grad=False`)**, so the function is exactly gamma-only LayerNorm eps 1e-5. Trainable parameter count equals SPEC exactly (S 162,417,408; F 407,914,496); the frozen betas add 19,200 (S) / 18,432 (F) parameters to `model_params` (0.01% / 0.005%). (v1 trained the beta, +19,200 trainable.)
- **No linear bias**: `add_bias_linear=False`; so "fused bias-GeLU" has nothing to fuse: the activation is TE's GELU op, which is the tanh approximation, identical to `F.gelu(approximate='tanh')`.
- QKV weight layout is re-interleaved per head for Megatron when loading the golden init, as in v1. Initial loss matches PyTorch (S: 11.71 at step 0 for batch 0; golden step-20 loss 8.57 vs PyTorch 8.56 in a 21-step check).
- Deviations/caveats: attention is cuDNN fused (not flash-attn) by TE's auto choice; fp32 main params are Megatron's own; gradient_accumulation_fusion off (see above).

## Rerun
```
./run_v2.sh                          # all of v2 (PHASES="count train_te train_pt_f train_graph serve", REPS=3); every job wrapped in flock ~/.gpu.lock via one_run.sh
TOTAL_SMS=148 TRAIN_SMS=116 INFER_SMS=24 ./run_v2.sh   # B200 split (needs a TE/torch build for sm_100)
python aggregate_v2.py               # -> results/baseline_v2_a100_<date>.json
MODEL=F python train_pytorch.py --golden      # F golden (golden/F/)
```
Single runs: `MODEL={S,F} python megatron_train.py --impl te [--attn flash] [--cuda_graph] [--count3] --out r.json`, `MODEL={S,F} python train_pytorch.py [--cuda_graph] [--count3] --out r.json`, `python serve_vllm.py HF_DIR [--count_launches --cg {default,eager,full}]`.
`--cuda_graph` (PyTorch) replays the whole step (gather, LR, fwd, bwd, AdamW; first 3 steps eager) from one graph; for Megatron only fwd+bwd is graphed (optimizer eager).

## Results (A100-40GB, n=3, mean +- sd, 4,000 steps x 8x512 tokens; SM-restricted = MPS 79% (84 SMs) for training, 16% (16 SMs) for vLLM)
| system | model | full GPU | SM-restricted | final loss (step 4000, full) |
|---|---|---|---|---|
| Megatron-core + TE (eager) | S | 209.2 +- 3.8 s, 78.3k +- 1.4k tok/s | 233.2 +- 6.3 s, 70.3k +- 1.9k tok/s | 2.130 |
| Megatron-core + TE (eager) | F | 265.2 +- 0.5 s, 61.8k +- 0.1k tok/s | 309.1 +- 1.1 s, 53.0k +- 0.2k tok/s | 2.518 |
| PyTorch eager | S (v1) | 207.9 +- 0.4 s, 78.8k tok/s | 226.2 +- 11.0 s, 72.5k +- 3.5k tok/s | 2.142 |
| PyTorch eager | F | 306.7 +- 0.1 s, 53.4k +- 0.0k tok/s | 352.9 +- 0.6 s, 46.4k +- 0.1k tok/s | 2.634 |
| PyTorch, whole step in ONE CUDA graph | S | 185.1 +- 0.1 s, 88.5k tok/s | not run | 2.145 |
| PyTorch, whole step in ONE CUDA graph | F | 304.6 +- 0.1 s, 53.8k tok/s | not run | 2.678 |
| Megatron-TE, fwd+bwd graph (optimizer eager) | S | 135.9 +- 0.1 s, 120.5k tok/s | not run | 2.130 |
| Megatron-TE, fwd+bwd graph (optimizer eager) | F | 263.2 +- 0.5 s, 62.2k tok/s | not run | 2.518 |
| Megatron-TE with flash-attn 2.8.3 (n=1) | S / F | 193.6 s / 265.4 s | not run | 2.138 / 2.570 |
| vLLM decode, batch 1, 256 tok | S | 523 +- 3 tok/s | 516 +- 11 tok/s | |
| vLLM decode | F | 539 +- 3 tok/s | 543 +- 3 tok/s | |
FA (B200): training 70.1 s (~230k tok/s), serving 710 tok/s, final loss 2.1094. Std is across the 3 reps (decode: 3 reps x 1 generate).
Findings: TE does not make Megatron faster than eager PyTorch for S (209 vs 208 s): at 12 layers and 4k tokens/step the step is host/launch-bound, as the graph variant shows (136 s with fwd+bwd in one graph, 185 s for PyTorch whole-step). F (4 layers, d=2048) is GEMM-bound, graphs buy ~1%.
Neither model gets near FA's 70 s on this A100 (108 SMs, different GPU from their B200, so absolute times are not comparable); F's 408M parameters are slower per token than S.
vLLM decode is launch/CPU-bound for both models (S 129 kernels/token, F 57), nearly SM-insensitive.

## Launch counts (details and the arithmetic in `launch_count_notes.md`)
Per step: Megatron-TE S 589/589/99, F 277/277/75; PyTorch S 580/580/1, F 229/229/1 under (a) kernels / (b) host calls / (c) host calls with the step in one CUDA graph (Megatron: fwd+bwd only).
Per token (vLLM): 129 (S) / 57 (F) kernels, 20 host calls with graphs on (floor 1). Whole demo with FA's ~49.6k served tokens:
| | (a) kernels | (b) host calls | (c) graphed | (c) graphed, 1 launch/token | FA |
|---|---|---|---|---|---|
| Megatron-TE S | 8.76M | 3.35M | 1.39M | 0.45M | 514,562 |
| Megatron-TE F | 3.94M | 2.10M | 1.30M | 0.35M | 514,562 |
| PyTorch eager S / F | 8.72M / 3.75M | 3.32M / 1.91M | 1.00M | 0.05M | |
**(c), with a graphed decode, lands closest** (0.7x-0.9x FA; 2.5x-2.7x with vLLM's real 20 launches/token); (a) and (b) are 4x-17x too high. FA's number needs <= ~128 launches per training step, so the demo's model is shallow and its stack fused/graphed.

## Caveats
- The GPU was shared with other agents; every job ran under the shared lock and `nvidia-smi` util was <5% at start for all runs (`noisy_not_idle_before_start`=0). `max_compute_procs_seen` is 1 (2-3 where MPS server or an unlocked process briefly appeared; see JSON); SM-restricted S sd is larger (MPS contention).
- F final loss differs between implementations (Megatron-TE 2.518 vs PyTorch 2.634 vs PyTorch-graph 2.678 single-step values; last-25 means 2.533/2.654/2.703), consistently across 3 reps; S agrees within 0.01. We did not find the cause for F (same data, init, schedule; candidates: fp32 vs bf16 reductions, Megatron fused CE, graph vs eager numerics). The F PyTorch/graph runs are equally valid timing baselines; do not use F loss to validate the megakernel to better than ~0.1.
- Megatron graph variant graphs fwd+bwd only (optimizer host-synchronises); PyTorch graph variant is the full step. Flash-attn rows have n=1.
- vLLM F/S serve the step-4000 weights of the matching PyTorch eager run; vLLM `cg full` equals its default (FULL_AND_PIECEWISE already uses a full decode graph).
- Raw per-run logs `results/runs_v2/*.log`; driver output `results/run_v2.out`.
