# Plan: match the Flapping Airplanes demo as closely as possible

## Every benchmark the video publishes, and how we reproduce each one

| # | They publish (B200, 148 SMs) | We reproduce it with | Status |
|---|---|---|---|
| 1 | Kernel launches: **1 vs 514,562** for the whole run | count the baseline's real CUDA kernel launches (CUPTI / torch.profiler) for 4,000 Megatron steps + vLLM decoding for the same wall time; megakernel = 1 | baselines |
| 2 | **4,000 training steps: 46.1 s vs 70.1 s** (baseline 34% slower) | same model/batch/optimizer in PyTorch+Megatron, timed alone, vs the in-kernel trainer running next to serving + Doom | baselines + integration |
| 3 | Training throughput **~355k vs ~230k tok/s** | tok/s from the same runs (4,096 tokens per step) | same |
| 4 | Inference **1,076 vs 710 tok/s** (34% slower) | vLLM batch-1 decode alone vs in-kernel decode in its SM partition | baselines + decode |
| 5 | **SM split 116 / 24 / 8 of 148**, baselines labelled "Megatron 116 SMs", "vLLM 24 SMs" (restricted to the same share) | proportional A100 split **85 / 17 / 6 of 108**; baselines restricted with CUDA MPS or green contexts | baselines, scheduler (integration) |
| 6 | Loss curve **11.5 -> 2.11** at step 4,000, baseline on the same curve | cl100k vocab (ln 100,277 = 11.51), TinyStories, identical init, batches, optimizer; overlay both curves | baseline + integration |
| 7 | Served text from the live model ("What is a megakernel?") | decode from the live bf16 weights, same prompt, sample at several training steps | decode + integration |
| 8 | Doom playing throughout, 8 SMs | already running in the kernel (35 fps, real engine); restrict it to its 6-SM share | done, repartition in integration |
| 9 | Everything in **46 s on 1 GPU, 1 launch** | one persistent launch, total wall time reported | integration |

## The model they imply (we use the same shape class)
Loss starts at ln(vocab) = 11.5, so the vocabulary is about 100k (cl100k_base = 100,277). 355k tok/s over 116 SMs of a B200 works out to a ~300M-parameter transformer at 4,096 tokens per step. We start with a 162M GPT (12 layers, d=768) and scale to ~300M (16 layers, d=1024) once the kernels are verified.

## Do we need a B200 demo?
- **For the ratios, no.** The method (individually timed baselines restricted to the same SM share, launch counts, loss-curve parity, 4,000-step time) can be run end to end on the A100 with a proportional 85/17/6 split.
- **For absolute 1:1 numbers (46 s, 355k tok/s, 116/24/8 of 148), yes, but only a small slice of code.** The scheduler, Doom, the tasks' structure, the baselines, benchmarks and the site are shared. What differs is the tensor-core path: the A100 code uses `mma.sync` (works on Blackwell but leaves most of its tensor-core throughput unused, so a fair comparison against Megatron/cuBLAS on a B200 needs a `tcgen05`/TMA GEMM and attention backend).
- **How we do it:** all A100 pieces are templated on d, L, H, T, V and the SM split, and every script takes TRAIN_SMS / INFER_SMS / TOTAL_SMS. When a B200 box exists: (1) run the A100 code unchanged (PTX JIT) for a first end-to-end number, (2) then write the sm_100 GEMM/attention backend with the hardware in front of us. Writing tcgen05 kernels blind, without a B200 to test on, would not give trustworthy code.
