# attention: causal MHA fwd + bwd tasks (bf16, dh=64)

Header-only `attention.cuh` (namespace `attn`), templates `<int T, int H>` (d = H*64). Verified at B=8, T=512, H=12 and T=1024.
mma.sync.m16n8k16 bf16, ldmatrix (swizzled 64x64 smem tiles), cp.async double buffering, flash-style online softmax (fp32 stats, exp2 domain).
All tasks: called by all 256 threads of one block, no blockIdx/gridDim, caller passes 128B-aligned dynamic smem. Deterministic (no atomics; the two
per-tile warp groups are combined in a fixed order). Tile = 64 rows (128-row tiles not done).

## Layouts
- `qkv [N,3d]` bf16, cols `[q|k|v]`, head h at cols `h*64..` of each third; token = b*T+t. `o`, `dout [N,d]` bf16 (heads concatenated).
- `lse [B,H,T]` fp32 natural log. `Dr [B,H,T]` fp32 = rowsum(dO*O). `dqkv [N,3d]` bf16, same column layout as qkv (dq|dk|dv).
- Scale 1/sqrt(64) is applied inside (dq, dk include it).

## Tasks (index = (b, h, tile), tile in [0, T/64); count B*H*T/64 each = 768 at the model shape)
| function | smem | needs complete | writes |
|---|---|---|---|
| `attn_fwd_tile<T,H>(qkv,o,lse,b,h,qt,smem)` | `SMEM_FWD`=40960 | - | o, lse |
| `attn_bwd_prep_tile<T,H>(o,dout,Dr,b,h,rt)` | 0 | fwd (all rows of that tile) | Dr |
| `attn_bwd_dkdv_tile<T,H>(qkv,dout,lse,Dr,dqkv,b,h,kt,smem)` | `SMEM_DKDV`=50176 | fwd + prep | dqkv dk, dv columns |
| `attn_bwd_dq_tile<T,H>(qkv,dout,lse,Dr,dqkv,b,h,qt,smem)` | `SMEM_DQ`=49152 | fwd + prep | dqkv dq columns |

Phases: fwd -> prep -> {dkdv, dq} (dkdv and dq are independent of each other and write disjoint columns, can share one queue).
Finer deps: fwd tile (b,h,qt) needs nothing; prep (b,h,rt) needs fwd tile rt only; dq(b,h,qt) needs fwd/prep tiles 0..qt (its lse/D rows are tile qt, o of others not needed) -> only lse+D of tile qt, plus qkv; dkdv(b,h,kt) needs lse/D of tiles kt..T/64-1.
Cost per task grows with position: fwd/dq tile qt does qt+1 key blocks; dkdv tile kt does T/64-kt query blocks. Schedule heavy tiles first (qt high / kt low).
Registers: 125-128 per thread (all heavy kernels hit the 128 cap of `__launch_bounds__(256,2)`; dkdv spills 4 bytes). Two blocks/SM fit regs and smem.
Work split inside a block: 8 warps = 4 row-groups of 16 rows x 2 halves (fwd/dq: key halves of each 64-key block; dkdv: query halves of each 64-row block), halves merged via smem at the end.

## Tests
`nvcc -arch=sm_80 -O3 -std=c++17 -o attn_test test_attention.cu && ./attn_test && python3 ref_attention.py` (add `-DATT_T=1024` for T=1024; `-DATT_B/-DATT_H` available).
`attn_test` runs grid kernels, a persistent-style run (216 blocks pulling scrambled tile indices from an atomic counter, dkdv+dq mixed in one queue) and checks bit-identity (PASS), dumps bins, times. `ref_attention.py` compares to fp32 SDPA (causal, autograd) and times PyTorch flash (bf16).

## Measured (A100, shared GPU; timings noisy, others were running at times)
Accuracy (error / max|ref|; mean err / mean|ref|), T=512: o 2.6e-3 / 1.9e-3, lse 1.3e-7, dq 4.4e-3 / 2.3e-3, dk 5.6e-3 / 2.3e-3, dv 2.9e-3 / 2.2e-3. T=1024 similar. PASS (bars 2e-2 / 5e-2).

| B=8,H=12 | ours fwd ms (TF/s) | torch flash fwd | ours bwd (prep+dkdv+dq) | torch flash bwd |
|---|---|---|---|---|
| T=512 (quiet GPU) | 0.060 (53.7) | 0.0525 (61.3) | 0.182 (44.3) | 0.283 (28.4) |
| T=512 (rerun, GPU busy) | 0.047 (67-69) | - | 0.146 (54-55) | - |
| T=1024 | 0.176 (73.1) | 0.163 (79.0) | 0.523 (61.6) | 0.760 (42.4) |
TF/s uses 4*B*H*T^2*64/2 for fwd, 2.5x for bwd. Torch bwd = (fwd+bwd) - fwd. Our timings are back-to-back kernel launches (launch gaps included) of the three bwd phases.
So fwd is within ~10-15% of flash, bwd ~1.5x faster than PyTorch's flash backward at this shape.

## Limits
- T multiple of 64, dh=64 only, causal only, no dropout. Index math in 32-bit for rows inside a tile, size_t for offsets.
- Output stores are 4-byte bf16x2 (not fully coalesced); fwd q-tile of 64 rows only.
- Dynamic smem > 48 KB needs `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` for the host-launched kernel.

## B200 notes (describe only)
On Blackwell this would be redone with tcgen05/UMMA with accumulators in TMEM, TMA loads of K/V/Q/dO tiles with mbarrier pipelines, 128-row query tiles (one CTA-pair/warpgroup per tile), and warp-specialized producer/MMA/softmax roles (FA4-style); bigger smem (up to ~227 KB) allows deeper K/V staging and keeping dK/dV accumulators in TMEM instead of registers. Not implemented here.
