# gemm2: v2 of the bf16 tensor-core GEMM tile library (sm_80)

Drop-in replacement for `llm/gemm/gemm.cuh`: `#include "gemm2.cuh"`, same namespace `gemm`, same `Args`, `Layout` (NN/NT/TN), `Epi`
(EPI_BF16/RES/GELU/DGELU/F32), same `gemm_tile<LAYOUT,EPI>(p, tile_m, tile_n, smem)` (128x128 tile, 256 threads, 64 KB dynamic smem, 2 blocks/SM,
<=128 regs), same `smem_bytes()`, `args_ok()`. No `blockIdx`, no atomics, deterministic (fixed K order), persistent-order bit-identical (tested).
Files: `gemm2.cuh`, `test_gemm2.cu` (extends test_gemm.cu: every case vs fp32 reference + cuBLAS + shuffled persistent bit-identity + RES aliasing; `-DCFG=n`
selects the tile config, `--steady` = wave-friendly mainloop-bound shapes), `run.sh` (sync+build+correctness), `runlock.sh`/`bench_box.sh`/`sweep.sh` (timing under `~/.gpu.lock`).

## What changed vs v1 (all measured, see tables)
1. **BK=64 double-buffered main loop** (2 smem stages of 32 KB = 64 KB for 128x128; the other resident block hides the load latency). v1: BK=32 / 4 stages.
   Halves barriers and cp.async address work per MMA. Steady-state NT 195 -> ~229 TF, TN 172 -> 186 TF in the same-session A/B (noise on the shared box is ~+-10%).
2. **Epilogue staged through smem** (BF16/RES/GELU/DGELU): per-warp fp32 staging (conflict-free pitch), then coalesced 8-byte stores, aux (residual / pre-activation)
   read/written coalesced. `up_fwd` GELU 0.45 -> 0.70 of cuBLAS, `down_bwd_dX` 0.57 -> 0.63. EPI_F32 stays direct (a quad already writes a full 32 B sector).
3. Generic `Cfg<BM,BN,WARPS_M,WARPS_N,STAGES,BK>` core `gemm_tile_cfg<Cfg,L,E>`; presets `CfgCompat` (default), `CfgCompat32` (v1 style), `CfgSmallN` (128x64), `CfgSmallM` (64x128),
   `CfgBig` (128x256, 64x64 warp tiles), `CfgBig3`, `CfgBigT` (256x128), `CfgBig64s3` (BK=64, 3 stages).

## Big (64x64 warp tile, 1 block/SM) result: NOT worth it
Built and measured (`CfgBig*`, 238-244 regs, no spill, 96 KB or 144 KB smem): steady state 195 TF (BK=32,4 st), 202 (BK=64, 2 st, 96 KB), 217 (BK=64, 3 st, **144 KB**),
vs 195-229 for the 2-block compat kernel, and worse on short-K / output-heavy shapes (logits 160 vs 188 TF: with one block per SM nothing overlaps the 128x256 fp32
epilogue/prologue; dW shapes have 18-72 tiles). The smem-read argument holds on paper (ldmatrix traffic 75% -> 50% of tensor time) but the 1-block kernel
loses the latency hiding of the second block and all barrier/epilogue bubbles are exposed. Spreading B-fragment loads over the k-step was slower (-8%).
So the megakernel should keep 2 blocks/SM (216 persistent blocks); no scheduler/partition change is needed.

## Contract / API additions
* `gemm_tile_cfg<Cfg, LAYOUT, EPI>(p, tile_m, tile_n, smem)`; tile covers `Cfg::BM x Cfg::BN`; `Cfg::SMEM` bytes (64 KB compat, 48 KB small), `args_ok_cfg<Cfg>(p)`.
* Alignment/limits as v1 (M%BM, N%BN, K%BK (BK=64 default!: **K%64==0**; d=768/2048 etc fine), lda/ldb%8, ldc%4, 16 B aligned A/B). `ldc` need not equal N.
* Rasterisation is the caller's: tile_m fastest (`t % tilesM`) keeps the wide B operand L2-resident (logits/dwte).
* Small-tile variants (`CfgSmallN/M`, 8 warps of 32x32, 48 KB, ~110 regs) are for small-N / small-M outputs (Wo, down 768-wide, dW_o) where 128x128 gives too few tiles.
  Split-K for dW_o (768x768, K=4096) is still the caller's job (even 128x64 gives 72 tiles).

## Results
Model S, grid launch, A100-SXM4, ratio = cuBLAS time / ours measured in the SAME run. The box was shared (other jobs ran unlocked at times: absolute TF vary +-10-15%,
cuBLAS itself moved 209 -> 162 TF on qkv_fwd between two runs), so compare ratios, and treat 0.02-0.05 differences as noise.

| case | v1 TF (ratio) | v2 compat TF (ratio) | v2 small 128x64 TF (ratio) |
|---|---|---|---|
| qkv_fwd | 141 (0.67) | 126* (0.78) | 160 (0.76) |
| wo_fwd | 122 (0.67) | 96* (0.67) | 119 (0.64) |
| up_fwd GELU | 98 (0.45) | 121* (0.72) | 149 (0.69) |
| down_fwd | 156 (0.65) | 127* (0.68) | 146 (0.61) |
| logits | 188 (0.89) | 203 (0.96) | 161 (0.75) |
| down_bwd_dX DGELU | 113 (0.57) | 129 (0.65) | 117 (0.59) |
| up_bwd_dX | 154 (0.65) | 175 (0.74) | 145 (0.62) |
| wo_bwd_dX | 126 (0.69) | 161 (0.88) | 132 (0.73) |
| qkv_bwd_dX | 149 (0.65) | 170 (0.74) | 142 (0.62) |
| head_bwd_dX | 174 (0.65) | 188 (0.69) | 152 (0.56) |
| dW_qkv | 142 (0.65) | 149 (0.69) | 146 (0.66) |
| dW_o | 53 (0.35) | 56 (0.37) | **87 (0.58)** |
| dW_up | 118 (0.61) | 128 (0.66) | 127 (0.65) |
| dW_down | 118 (0.60) | 129 (0.66) | 123 (0.63) |
| dwte | 147 (0.58) | 147 (0.57) | 79 (0.29) |
(*) that compat run overlapped with a foreign job (cuBLAS reference for those rows also dropped ~22%); use the ratio column.

Steady state (wave-exact shapes 2048x6912x2048, same lock hold, `--steady`): v1-style 4-stage BK=32 compat 195 TF (NT) / 195 (NN) / 173 (TN F32) vs cuBLAS 262-268 / 219;
v2 compat BK=64 double-buffered: **229 (NT, 0.87) / 201 (NN, 0.76) / 186 (TN, 0.85)**, GELU K=768 164 (0.73). 128x256 variants: 195 / 202 (BK=64 2 st) / 217 (BK=64 3 st, 144 KB).

Resources (compat default): 128 regs (cap), 64 KB dyn smem, no spill in the EPI_F32 kernels; the smem-staged epilogue kernels spill 8-28 B/thread (epilogue only, L1 hits).
Small configs: ~105-112 regs, 48 KB. Big: 234-244 regs, 96 KB (144 KB for BK=64/3 st).

## Goal check
>= 0.85 of cuBLAS was reached only on logits (0.96, cuBLAS is weak on that shape), wo_bwd_dX (0.88) and steady-state NT/TN (0.87/0.85). Typical big shapes are 0.65-0.77
(head_bwd_dX 0.69, dwte 0.57). dwte is bound by the TN (MN-contiguous A and B, ldmatrix.trans) + fp32 read-modify-write C epilogue and 1.6 GB of C traffic; head_bwd_dX (96..192 tiles, K=100352) is
wave/tail limited without split-K. Not reached: >=0.85 across the board; the remaining gap to cuBLAS (~15-25%) is main-loop efficiency of mma.sync with 64x32 warp tiles at 128 regs.

## What the megakernel must change
Nothing structural: swap the include (`gemm2.cuh`), keep 2 blocks/SM, 64 KB smem, same call. Only requirement beyond v1: **K % 64 == 0** (all model S/M K values satisfy it).
Optional: use `CfgSmallN` (tile 128x64: tile_n counts in 64-wide units) for dW_o-like outputs; use split-K for dW_o/head_bwd_dX; keep tile_m-fastest ordering for wide-B GEMMs.
Open issues: GPU contention made timing noisy; no ncu on the box so the bottleneck analysis is from A/B experiments (no-ldmatrix-spread, BK, stages), not counters;
epilogue-prefetch of the next tile's pipeline (overlapping epilogue with next prologue) was not implemented.

