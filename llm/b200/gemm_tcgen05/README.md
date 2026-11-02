# gemm_tcgen05: Blackwell GEMM backend (TMA + tcgen05.mma + TMEM), first attempt

**Status (2026-11-02, B200):** `test_gemm_sm100` prints PASS for all 15 S shapes, every layout, persistent == grid bit-exact, at 34-73%
of cuBLAS ([`../../../b200-run/results/gemm_v2/`](../../../b200-run/results/gemm_v2/); the `mma.sync` control: 13-33%). The B200 megakernel
builds with it (`-DTRIAD_GEMM_BACKEND=1`) and trains about 2x faster than with `mma.sync`
([`../../../b200-run/README.md`](../../../b200-run/README.md#the-gemms-from-mmasync-to-tcgen05)). It was first written without B200 hardware;
what the first runs found and fixed:

* **TMEM and blocks per SM.** The occupancy API reports 1 block per SM for any kernel that contains a `tcgen05` instruction, and a block
  that holds its TMEM allocation permit makes the next block's `tcgen05.alloc` on that SM wait until the holder exits. So each block
  allocates `TMEM_COLS` columns once in `block_init()` (all 256 threads, before its first tile), relinquishes the permit immediately, and
  frees them in `block_exit()`; `tile()` never allocates. Callers size their grid from registers and shared memory (2 blocks per SM fit).
  A per-tile allocation without the relinquish serializes co-resident blocks, and in a persistent kernel can deadlock them.
* **Epilogue.** The accumulator goes TMEM -> registers -> a 64 KB fp32 staging tile in the idle pipeline buffers (16-B chunks XOR-
  swizzled by row) -> one tile row per warp store; the per-element math is `gemm::epi_store`'s, so the results are bit-identical. This
  roughly doubled the K=768 GEMMs (qkv 300 -> 764 TFLOP/s).
* **Shared memory.** The mbarriers live in static shared memory and the alignment slack is 896 B: `smem_bytes()` is 66,432 B for 2
  stages, 99,200 B for 3 (`-DGEMM100_STAGES`).
* The `tcgen05` GEMMs round exactly like the `mma.sync` ones on every shape we ran: the megakernel's loss is bit-identical with either.

## Files
| file | what |
|---|---|
| `gemm_sm100.cuh` | the backend. `gemm::sm100::tile<LAYOUT,EPI>(ArgsTC, tile_m, tile_n, smem)` + host `prepare()` (TMA descriptors) |
| `gemm_backend.cuh` | selects backend with `-DTRIAD_GEMM_BACKEND=0|1`; uniform `tg::` interface used by `../megakernel/triad.cu` |
| `make_test.sh` -> `test_gemm_sm100.cu` | derives the test from `llm/gemm/test_gemm.cu` (same 15 S cases, `--M` for 15 M cases, fp32 reference, cuBLAS compare + timing, persistent-vs-grid bit-exactness, RES aliasing) |

## Interface vs llm/gemm/gemm.cuh
Same task contract: one 128x128 output tile per call, all 256 threads of the block call it, no blockIdx/gridDim, `__syncthreads()` at the end, same `Layout` (NN/NT/TN),
same `Epi` (BF16, RES, GELU, DGELU, F32 with accumulate) and the same epilogue code (`gemm::epi_store`, reused). Differences:
1. `Args` becomes `ArgsTC` (= `gemm::Args` + `tmA`, `tmB`). Call `prepare(args, layout)` once on the host (the megakernel does it inside `mkgemm`).
2. `K % 64 == 0` (all S/F/M shapes satisfy it: 768, 2048, 3072, 4096, 8192, 100352).
3. `smem_bytes()` = stages x 32 KB + 896 B of alignment slack (66,432 B for 2 stages, 99,200 B for 3); barriers in static shared memory.
4. Each block holds 128 of the 512 TMEM columns from `block_init()` to `block_exit()` (2 blocks per SM => 256 columns).
5. Compiled for any target without `__CUDA_ARCH_FEAT_SM100_ALL` (sm_80, sm_90, plain sm_100) the same source silently becomes the mma.sync GEMM, so a TRIAD_GEMM_BACKEND=1 binary still builds and runs on the A100 (it just is not tcgen05; checked: test builds for sm_80).

## Design (single CTA, cta_group::1, M=128, N=128, K=16 per instruction, 4 MMAs per 64-wide K stage)
* Per stage: A tile 16 KB + B tile 16 KB, `SWIZZLE_128B`, K-chunk BK=64 bf16 (= 128 B rows). 3-stage mbarrier pipeline (`full[s]`, `empty[s]`), plus one `accum` barrier.
* Warp 0 lane 0: TMA producer (`cp.async.bulk.tensor.2d ... mbarrier::complete_tx::bytes`). Warp 1 lane 0: MMA issuer
  (`tcgen05.mma.cta_group::1.kind::f16 [tmem], a_desc, b_desc, idesc, p`; `tcgen05.commit ... mbarrier::arrive::one` frees the stage and, at the end, signals `accum`).
  All 8 warps: epilogue. Warp w reads TMEM lane quadrant `w&3` and column half `w>>2` with `tcgen05.ld.32x32b.x32` (thread = one row, 64 columns) and calls the old epilogue.
* Operand majors: K-major = NN's A, NT's A and B; MN-major = NN's B, TN's A and B. MN-major operands are loaded as two 64-wide TMA boxes (8 KB each) and described with `a_major/b_major=1` in the instruction descriptor.
* Accumulation order over K is fixed (deterministic), as in the A100 library.

## Mirrored CUTLASS / reference files (from memory; open them on the B200 box and diff against `gemm_sm100.cuh` first thing if the test fails)
`include/cute/arch/mma_sm100_umma.hpp` (SM100_MMA_F16BF16_SS), `include/cute/arch/mma_sm100_desc.hpp` (SmemDescriptor, InstrDescriptor, LayoutType),
`include/cute/atom/mma_traits_sm100.hpp` (make_umma_desc: LBO/SBO rules), `include/cute/arch/copy_sm100.hpp` (tcgen05.ld), `include/cute/arch/tmem_allocator_sm100.hpp`,
`include/cutlass/arch/barrier.h` (mbarrier, umma_arrive), `include/cute/arch/copy_sm90_tma.hpp` (TMA 2D), `include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp` (pipeline structure),
`examples/70_blackwell_gemm/` (smallest end-to-end reference), and DeepGEMM's sm100 kernels for the "advance the descriptor start address by 32 B per k16 inside a 128-B swizzle atom" idiom.

## Known weak points, ranked by how likely they are to be wrong (R1 is the most likely)
* **R1 MN-major descriptors** (NN's B operand = every forward linear!, TN's both): LBO=8192 (stride between the two 64-wide MN halves), SBO=1024 (8-K-row groups), k16 step = +2048 B.
  I am least sure about which of LBO/SBO plays which role for MN-major SW128 and whether the k16 step is +2 groups. Symptom: NN/TN cases fail, NT passes. Debug: `-DGEMM100_LAYOUT_MASK=2` (only NT via tcgen05, others mma.sync) and compare `make_umma_desc` in `mma_traits_sm100.hpp`.
* **R2 K-major descriptors** (NT, and A of NN): LBO field = 16 B ("unused"), SBO = 1024, +32 B per k16. If NT fails with garbage in a stripe pattern, check SBO and the version bit (bit 46 = 1) first.
* **R3 mbarrier parities / commit semantics**: `empty` wait uses `(use&1)^1` (first pass falls through), `full` `use&1`, `accum` parity 0. If the kernel hangs (test prints nothing and the GPU stays at 100%), this is the suspect; run `compute-sanitizer --tool synccheck` and shrink to K=64.
* **R4 proxy fences**: the megakernel's earlier phases write operands with the generic proxy; the TMA reads them through the async proxy. The producer executes `fence.proxy.async` once per tile; if stale data shows up only inside the megakernel (and the stand-alone test passes), add `fence.proxy.async.global` after the dependency acquire in `job_loop` and run with `--opt-mode 0`.
* **R5 epilogue speed**: done (staged, coalesced; see the status above). A TMA store would also free the warps earlier.
* **R6 TMEM alloc inside a tile**: done (once per block with the permit relinquished; this was what held the megakernel at 1 block per SM).
* Not implemented: 2-CTA (cta_group::2, 256x128 per CTA pair), FP8/NVFP4 kinds, multicast TMA, persistent tile scheduler with accumulator double-buffering (epilogue overlap with the next tile's MMAs). Those are the tuning list for hour 4.

## Build and test plan (reuses test_gemm.cu through make_test.sh; run on the B200)
```
cd llm/b200/gemm_tcgen05 && ./make_test.sh
# 0. backend 0 control: the old mma.sync GEMM through the same harness (must PASS; baseline TFLOP/s on B200)
nvcc -arch=sm_100a -O3 -std=c++17 -DTRIAD_GEMM_BACKEND=0 -Xptxas -v -o t_mma  test_gemm_sm100.cu -lcublas && ./t_mma
# 1. tcgen05, NT only (both operands K-major; all other layouts fall back to mma.sync). Start small:
nvcc -arch=sm_100a -O3 -std=c++17 -DTRIAD_GEMM_BACKEND=1 -DGEMM100_LAYOUT_MASK=2 -o t_nt test_gemm_sm100.cu -lcublas
timeout 60 ./t_nt --only logits --nobench          # a hang here = R3
timeout 120 ./t_nt --nobench                       # all 15 cases; NT cases exercise tcgen05
# 2. add NN (mask 3), then TN (mask 7 = default)
nvcc ... -DGEMM100_LAYOUT_MASK=3 ... ; then default
# 3. performance vs cuBLAS (the number that matters), the S and M shapes
./t_all ; ./t_all --M
# 4. compute-sanitizer for races/illegal TMA coords:   compute-sanitizer --tool memcheck ./t_all --only qkv_fwd --nobench
```
PASS criteria are the A100 ones: bf16 outputs relmax <= 2e-2 (activations) / 5e-2 (gradients; measured 2.6e-3 on A100), fp32 outputs ~1e-6, persistent==grid **bit-exact**, RES aliasing OK.
Note the test links nothing from the driver (`cuTensorMapEncodeTiled` is fetched through `cudaGetDriverEntryPoint`).

## Fallback switch to the existing mma.sync path
* Whole backend: `-DTRIAD_GEMM_BACKEND=0` (the default). It runs on B200 when compiled with `-arch=sm_100a` (mma.sync is fully supported, only slow relative to tcgen05).
* Per layout: `-DGEMM100_LAYOUT_MASK=<bits>` bit0 NN, bit1 NT, bit2 TN. Layouts not in the mask call the original `gemm::gemm_tile` inside the same kernel.
* Automatic: any compile target without `sm_100a` features uses mma.sync.
