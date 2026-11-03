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
Caveat: the GPU was shared with other jobs; some runs saw 2-3 compute processes on the GPU (see `max_compute_procs_seen_per_run`), which includes our own
process/MPS server, so the figures are probably but not provably clean. Megatron without TE is much slower than eager PyTorch here (unfused attention), so it is a weaker baseline than FA's (which presumably used TE).
vLLM decode is CPU/launch-bound at this model size, hence almost insensitive to SM count.
