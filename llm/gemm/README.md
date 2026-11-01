# gemm: bf16 tensor-core GEMM tile library (sm_80)

Header-only `gemm.cuh`. One **task = one 128x128 output tile**, executed by all 256 threads of a block
(8 warps as 2x4, warp tile 64x32), `mma.sync.m16n8k16` bf16 -> fp32, `ldmatrix`(.trans), 4-stage `cp.async` pipeline,
BK=32, XOR-swizzled bank-conflict-free smem. No `blockIdx`/`gridDim`, no atomics, no global barriers; deterministic
(fixed reduction order over K, each tile owns its output). Files: `gemm.cuh`, `test_gemm.cu`, `ref_gemm.py`, `run.sh`
(sync+build+run on the box), `bench.sh`, `out_S.txt` / `out_M.txt` (last full runs).

## API
```cpp
#include "gemm.cuh"
using namespace gemm;
Args p{};  p.A,p.B,p.C,p.aux, p.lda,p.ldb,p.ldc,p.ldaux, p.M,p.N,p.K, p.accumulate;
template<int LAYOUT /*NN|NT|TN*/, int EPI, int STAGES=4>
__device__ void gemm_tile(const Args& p, int tile_m, int tile_n, char* smem);   // all 256 threads call it
```
Output `C[M,N]` (row-major, `ldc`), reduction length `K`; tile `(tile_m, tile_n)` covers rows `128*tile_m..`, cols `128*tile_n..`.
Task count per GEMM = `(M/128)*(N/128)`; tasks of one GEMM are mutually independent. Inputs (A, B, aux for RES/DGELU) must be complete
(earlier phase) before the tile runs. `smem` = dynamic smem base, 128 B aligned, `smem_bytes<STAGES>()` = **64 KB** (4 stages; 3 stages = 48 KB, 5 = 80 KB,
compile with `-DGEMM_STAGES=n`). The function ends with `__syncthreads()`, so smem is reusable immediately. Host helper `args_ok(p)` checks the rules below.

| LAYOUT | A stored | B stored | use |
|---|---|---|---|
| NN | `[M,K]` (lda>=K) | `[K,N]` (ldb>=N) | forward `Y = X.W`; also `dX_head = dlogits[N,V] . wte[V,d]` (K=V) |
| NT | `[M,K]` | `[N,K]` (ldb>=K) | `dX = dY . W^T` with W stored `[N,K]` i.e. `[d_in,d_out]` (here GEMM-N = d_in, GEMM-K = d_out); **logits = x . wte^T** (wte `[V,d]`) |
| TN | `[K,M]` (lda>=M) | `[K,N]` | `dW = X^T . dY`: X `[tokens,d_in]`, dY `[tokens,d_out]`, K = tokens; `dwte += dlogits^T . x` (M=V, N=d, K=tokens) |

| EPI | result |
|---|---|
| `EPI_BF16` | `C = bf16(acc)` |
| `EPI_RES` | `C = bf16(acc + R)`, `R = aux` (bf16, `ldaux`); `C` may alias `aux` (each element read then written by the same thread) |
| `EPI_GELU` | `aux = bf16(acc)` (pre-activation, saved), `C = bf16(gelu_tanh(float(aux)))` (uses the rounded pre-act so fwd/bwd agree) |
| `EPI_DGELU` | `C = bf16(acc * gelu'(aux))`, `aux` = saved bf16 pre-activation (MLP-up backward `dH`) |
| `EPI_F32` | fp32 `C`: `accumulate ? C += acc : C = acc` (logits chunk: accumulate=0; dW32: accumulate=1, first micro-step 0) |

GELU/GELU' use `tanh.approx.f32` (rel. err ~2^-11, below bf16 rounding); `gelu_tanh`, `gelu_tanh_grad` are exported.

**Limits / rules**: M%128==0, N%128==0, K%32==0 (asserted only by host `args_ok`; device code does not check), `lda,ldb,ldaux%8==0`, `ldc%4==0`,
A/B/aux 16 B aligned, C 8 B aligned (4 B for bf16 C). `ldc/ldaux` need not equal N. Split-K is the caller's job (offset A/B, shorter K; with `EPI_F32`
sequential `accumulate=1` calls in different phases, or separate buffers) - **recommended for small-output TN/NT GEMMs**: `dW_o` (768x768) has only 36 tiles and
`head_bwd_dX` (4096x768, K=100352) only 192; both underfill 216 slots. Register budget: kernels compile at 124-128 regs, no spills (`__launch_bounds__(256,2)`);
if the megakernel is compiled for 1 block/SM with more registers nothing changes functionally. Tile rasterisation is the caller's choice (test uses `tile_m = t % tilesM` fastest, good for L2 reuse of the wide B operand).

## Verification (`test_gemm.cu`, build `nvcc -arch=sm_80 -O3 -std=c++17 -o gemm_test test_gemm.cu -lcublas`, run `./gemm_test [--M] [--only s] [--nobench] [--nopersist]`)
For every case: (1) ordinary grid launch vs an **fp32 reference kernel** (bf16 inputs exact in fp32, epilogue applied in fp32 with accurate `tanhf`) - reported
`relmax = max|err| / max|ref|` and `relmean = mean|err| / mean|ref|`; (2) cuBLAS (`cublasGemmEx`, bf16 in, fp32 compute) vs the same reference (`cublas_rm`, for context);
(3) **persistent kernel**, 2 blocks/SM, each block loops over a *shuffled* tile order and calls `gemm_tile`; output compared **bitwise** to the grid launch (also the saved pre-activation);
(4) `EPI_RES` with `C == aux` aliasing is bitwise equal to the non-aliased result. TN cases start from a random nonzero fp32 `C0` and test `accumulate=1`.
Bar: 2e-2 (activations) / 5e-2 (gradients); measured ~2.6e-3 for bf16 outputs (= bf16 rounding) and <5e-6 for fp32 outputs. `ref_gemm.py` cross-checks the GELU/GELU' formulas vs PyTorch (diff 1e-7).
**Result: all 15 S cases and all 15 M cases PASS, persistent == grid bit-exact, alias OK.**

## Performance, model S (A100-SXM4, grid launch, best of 6x(5..20) runs; same box was shared, so absolute TF vary +-15% with other jobs)
| case | layout/epi | M x N x K | ours TF | cuBLAS TF | ratio | relmax |
|---|---|---|---|---|---|---|
| qkv_fwd | NN BF16 | 4096x2304x768 | 121 | 180 | 0.67 | 2.6e-3 |
| wo_fwd | NN RES | 4096x768x768 | 100 | 150 | 0.67 | 2.2e-3 |
| up_fwd | NN GELU | 4096x3072x768 | 84 | 177 | 0.48 | 2.6e-3 |
| down_fwd | NN RES | 4096x768x3072 | 128 | 195 | 0.66 | 2.2e-3 |
| logits | NT F32 | 4096x100352x768 | 166 | 224 | 0.74 | 1.8e-6 |
| down_bwd_dX | NT DGELU | 4096x3072x768 | 116 | 198 | 0.58 | 2.4e-3 |
| up_bwd_dX | NT RES | 4096x768x3072 | 156 | 238 | 0.65 | 2.2e-3 |
| wo_bwd_dX | NT BF16 | 4096x768x768 | 126 | 183 | 0.69 | 2.6e-3 |
| qkv_bwd_dX | NT RES | 4096x768x2304 | 149 | 229 | 0.65 | 2.5e-3 |
| head_bwd_dX | NN BF16 | 4096x768x100352 | 174 | 272 | 0.64 | 3.3e-3 |
| dW_qkv | TN F32 | 768x2304x4096 | 144 | 222 | 0.65 | 3.9e-6 |
| dW_o | TN F32 | 768x768x4096 | 53 | 151 | 0.35 | 3.7e-6 |
| dW_up | TN F32 | 768x3072x4096 | 118 | 194 | 0.61 | 4.4e-6 |
| dW_down | TN F32 | 3072x768x4096 | 118 | 197 | 0.60 | 3.9e-6 |
| dwte | TN F32 | 100352x768x4096 | 146 | 259 | 0.56 | 4.7e-6 |

Model M (d=1024, h=4096; see `out_M.txt`, taken while the GPU was shared so cuBLAS numbers are depressed): same PASS pattern, ours 120-195 TF
(qkv 154, up 124 (GELU), down 147, logits 195, dW_qkv 154, dwte 147; ratios 0.5-0.9, noisy).
Steady-state ceiling of this kernel is ~175 TF (56% of the 312 peak) vs cuBLAS 270-280: **the >=70%-of-cuBLAS goal is NOT met on most shapes** (0.56-0.74; 0.35/0.48 for the outliers below).

Why / what was tried:
* Main-loop limit: with 8 warps of 64x32 each k16 step needs 6 `ldmatrix.x4` per 16 MMAs; smem read time (~192 clk per block-k16-step) is ~75% of tensor time (256 clk), so the SM cannot saturate the tensor cores. A 64x64 warp tile (128x256 block tile) would fix it but needs >128 regs/thread (1 block/SM), which conflicts with the 2-blocks/SM contract; not built.
* Done: fine-grained software pipeline (A frags double-buffered per 16 rows, next k-step/next-stage frags prefetched during MMAs, refill of the freed smem slot after the mid-iteration barrier), non-volatile MMA asm, stage count sweep (3 stages clearly worse, 4 best, 5 not better). A first coarse double-buffering attempt spilled registers and was slower; discarded.
* Outliers: `up_fwd`/`down_bwd_dX` (0.48/0.58) pay for the GELU/GELU' epilogue (SFU `tanh` + 2 bf16 streams) on a short K=768 mainloop; epilogue stores go straight from the accumulator fragments (not smem-staged). `dW_o` has only 36 tiles (see split-K above).
* Wave quantisation (e.g. 576 tiles over 216 slots) costs up to ~10% on grid launches; in the megakernel the scheduler can fill those slots with other tasks.

## B200 notes (not testable here)
On sm_100 the same tile contract would be re-implemented with `tcgen05.mma` (5th-gen tensor cores): accumulators live in Tensor Memory (TMEM, 256 KB/SM) instead of registers, so large tiles (128x256, even 256x256 with CTA pairs `cta_group::2`) no longer cost registers and the register-limited warp-tile problem above disappears; A/B are read straight from swizzled smem via descriptors (no `ldmatrix`), operands are loaded with TMA (`cp.async.bulk.tensor`, mbarrier-based multi-stage pipeline, multicast within a cluster) issued by a single elected thread, MMA is issued by one thread and completion tracked with mbarriers/`tcgen05.commit`, and the epilogue reads TMEM with `tcgen05.ld` (a warpgroup) then applies the same bias-free epilogues. Block-scaled FP8/FP4 (`mxf8`/`nvfp4`) become available. The megakernel task contract (one tile per call, smem budget, no blockIdx) would remain, but a task would likely be a warp-specialised group (TMA warp, MMA warp, epilogue warps) rather than 256 symmetric threads, and smem budget per task can be larger (up to ~227 KB).
