# Triad LLM: contract for the pieces that go into the megakernel

Goal: replace the toy char-level MLP with a real GPT-style transformer LM that trains (forward, backward, AdamW) and generates text INSIDE the persistent
megakernel, next to Doom, on one NVIDIA A100-SXM4-40GB (sm_80, 108 SMs, bf16 tensor cores via `mma.sync`, `cp.async`; NO Hopper/Blackwell features).
Each workstream below builds device functions ("tasks") that the megakernel scheduler will call; it also verifies them standalone.

## Model "S" (fixed; kernels should be shape-generic through constexpr config, but verified at these shapes)
- GPT, pre-LayerNorm, L=12 layers, d=768, H=12 heads (dh=64), context T=512, MLP hidden 4d=3072, GELU(tanh approx, same as `F.gelu(x, approximate='tanh')`).
- Vocab V=100352 (cl100k_base has 100277 tokens, padded up to a multiple of 128). Learned absolute positions wpe[T,d]. LM head TIED to token embedding wte[V,d].
- No bias on any linear layer. LayerNorm has gamma only (no beta), eps=1e-5 (`F.layer_norm(x, (d,), weight, None, 1e-5)`). Final LayerNorm before the head.
- Init: N(0, 0.02) for all matrices and embeddings; attention-proj and MLP-down weights scaled by 1/sqrt(2L); gamma = 1.
- ~162M parameters. A bigger "M" config (L=16, d=1024, H=16) will be used later, so keep d, L, H, T, V as constexpr template parameters.
- Training batch: B=8 sequences x T=512 = N=4096 tokens per step. Loss = mean token cross-entropy (fp32).

## Precision and layout
- Master weights fp32 (`W32`), working copy bf16 (`Wbf`), activations bf16, accumulation fp32, softmax statistics fp32, AdamW m/v fp32, gradients: bf16 activations-grads, fp32 weight-grad buffers.
- Everything row-major. Linear layers store W as [in, out] so forward is Y[N,out] = X[N,in] . W[in,out].
  Backward: dX = dY . W^T (an "NT" GEMM), dW = X^T . dY (a "TN" GEMM).
- Per-layer weights: Wqkv[d,3d] (columns = [q|k|v], each split into H heads of 64), Wo[d,d], Wup[d,4d], Wdown[4d,d]; gamma1[d], gamma2[d]; global: wte[V,d], wpe[T,d], gamma_f[d].

## How tasks are called (the megakernel contract)
- The megakernel runs persistent blocks of 256 threads. A "task" is a `__device__ __forceinline__` function (templates welcome) that all 256 threads of one block call together,
  with all inputs as arguments/pointers and a task index. It must NOT use blockIdx/gridDim/cooperative-groups grid sync/dynamic parallelism/global barriers. `__syncthreads()` inside is fine.
- Shared memory: dynamic, <= 80 KB per task (two blocks per SM) preferred; <= 96 KB acceptable if documented. The caller provides a `char*` smem base (aligned to 128 B).
- Tasks inside one phase must be independent of each other; the caller orders phases with dependency counters. For every function document: inputs, outputs, which earlier phases must be complete, task count and granularity.
- Prefer deterministic results (fixed reduction order, no atomics on floats). If something is not deterministic, say so.
- Ownership: put code in headers `llm/<area>/*.cuh` (header-only), a standalone test `llm/<area>/test_<area>.cu`, a PyTorch reference check `llm/<area>/ref_<area>.py` (torch 2.7 / CUDA 12.8 is installed on the box with plain `python3`), and `llm/<area>/README.md` with the API, shapes, measured accuracy and speed.
- Accuracy bar (bf16 aware): compare against an fp32 PyTorch reference computed from the same bf16 inputs. Activations/outputs relative error <= 2e-2, gradients <= 5e-2 (report max and mean). Print `PASS`/`FAIL`.
- Speed: report TFLOP/s (GEMM, attention) or GB/s (memory-bound ops) from running the tasks as an ordinary grid, and compare with cuBLAS / PyTorch where applicable.

## Environment and etiquette (IMPORTANT, 5 agents share ONE A100)
- Box: `ubuntu@<ip>` (ssh works without a password). nvcc 12.8, `-arch=sm_80`. Work in `~/llm/<area>/` on the box and keep the source of truth locally in `/Users/jhome/flapping-airplanes/llm/<area>/` (rsync it up; never edit only on the box).
- GPU is shared: keep each run short (< 60 s where possible). Correctness runs may happen any time. For TIMING runs, first check `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader` and retry a few times if another job is running; report noise honestly.
- Use uniquely named binaries prefixed with your area. Never run `pkill -f`; never kill processes you did not start; `pkill -x <your_binary>` only.
- Touch only your own `llm/<area>/` directory (and `~/llm/<area>/`, plus the baseline agent's venv/data locations). Do NOT modify live-demo/, doom-gpu/, benchmarks/, site/, tools/, README.md. Do NOT run any git command.
- No package installs or downloads unless your brief explicitly allows it (only the baseline workstream does).
- Be honest: report what failed, measured numbers (not hoped-for ones), and known limitations.
