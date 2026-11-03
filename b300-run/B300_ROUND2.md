# B300 round 2: tcgen05 GEMM + working tcgen05 megakernel

All numbers: Modal `gpu="B300"` (pinned; verified `NVIDIA B300 SXM6 AC`, cc 10.3, 148 SMs), driver 610.57, nvcc 13.0.48, binaries built `-gencode arch=compute_100f,code=sm_100f` (no PTX, so no silent JIT fallback). Driver: `modal/app2.py` (plan JSONs: CPU build container, then one GPU container). Raw logs: `results/r2_*`.

## 0. Corrections to round 1 (important)
* `gpu="B200+"` is **not** a fixed model: one round-2 call landed on a real **B200** (cc 10.0). The round-1 "sm_103a: no kernel image" was almost certainly a B200 (sm_103 cubins cannot load there). On the B300 `sm_103a`, `sm_103f`, `sm_100f` all run the tcgen05 test; **`sm_100a` does NOT load on the B300** ("no kernel image"), so use `sm_100f` (family target; runs on B200 and B300) or `sm_103a`. Round-1 mixed-hardware numbers should be treated with care; the baselines were re-measured here on a pinned B300 (section 4).
* A kernel that contains `tcgen05.alloc` gets **1 CTA/SM** (occupancy API and observed co-residency, 32 KB or 100 KB smem, 128/256/512 columns; `gemm_tcgen05/occ_probe.cu`). So the tcgen05 megakernel is 148 blocks x 256 threads, not 296. This costs the non-GEMM tasks and the decode job their second block.

## 1. Why the megakernel hung (fixed)
Root cause: **mbarriers re-initialised by the next task without `mbarrier.inval`** (plus stale async `tcgen05.commit` arrivals). The first GEMM tile of each block worked, the second hung waiting for the accumulator barrier (found with a clock64-timeout/trap debug build `-DG2_DEBUG`, `tiles_started=2` in every hang report; `--train-sms 1` reproduced it with one block). Fix in `tile_tc`: before the final `__syncthreads` thread 0 waits for the last arrival of every `empty[]` barrier and `mbarrier.inval`s all barriers. Also done: `__syncwarp()` before the `.sync.aligned` alloc/dealloc, TMEM allocated once per block (`G2_PERSIST_TMEM`, `tg::block_begin/end`). Ruled out: smem offset/alignment, proxy fences, TMEM re-alloc, register allocation, job-loop sequence.

## 2. GEMM design (`gemm_tcgen05/gemm_sm100_v2.cuh`, glue `gemm_backend_r2.cuh`, test `make_test_r2.sh -> test_gemm_r2.cu`)
Structure follows CUTLASS sm100 (`sm100_mma_warpspecialized.hpp`, `sm100_gemm_tma_warpspecialized.hpp`; cited from memory, not diffed):
* Warp roles inside one task: warp0/lane0 TMA producer, warp1/lane0 `tcgen05.mma` issuer, epilogue warps; 4-stage (wide) mbarrier TMA pipeline (full/empty), SW128, 1024-B aligned stages. **Warp specialisation is compatible with the megakernel contract** (all 256 threads still enter/leave together).
* Wide config: MMA 128x256, 4 stages x 48 KB. A task = `nsub` (1..4) consecutive 128x256 tiles with a **double-buffered TMEM accumulator**: epilogue of tile j (warps 2-7) overlaps the MMAs of tile j+1; last tile uses all 8 warps. Host policy: `nsub = ceil(tiles/train_slots)` (one wave). Narrow config (128x128) only for N not multiple of 256 (unused for S/F/M).
* Epilogue: TMEM -> regs (two `tcgen05.ld.x32` in flight) -> per-warp 4 KB XOR-swizzled smem transpose -> coalesced 16-B global accesses (64-128 B contiguous per row); aux/accumulate-C loads issued before the transpose. This was the biggest single win (qkv 0.37 -> 0.75 of cuBLAS). Epilogues are bit-identical in math to the old ones.
* Not done: 2-CTA (`cta_group::2`) - needs cluster launch and paired task claiming; the megakernel is a plain persistent grid with per-block ticket claims, so it cannot pair blocks without a scheduler redesign. TMA multicast same. Roofline note: single-CTA 128x256 tiles are L2->SM-bandwidth bound (~10 TB/s measured), which is why 2-CTA is what lets cuBLAS reach 1.3-1.7 PF.

### Per-case vs cuBLAS, same process, S shapes, fp32-reference + persistent bit-exact + RES-alias checks (all 15 PASS)
| case | ours TF | cuBLAS TF | ratio | cfg(wide*10+nsub) |
|---|---|---|---|---|
| qkv_fwd | 969 | 1303 | 0.74 | 12 |
| wo_fwd | 457 | 756 | 0.60 | 11 |
| up_fwd (GELU) | 808 | 1231 | 0.66 | 13 |
| down_fwd | 910 | 1347 | 0.68 | 11 |
| logits (F32 out) | 851 | 1304 | 0.65 | 14 |
| down_bwd_dX (DGELU) | 559 | 1315 | 0.43 | 13 |
| up_bwd_dX | 910 | 1345 | 0.68 | 11 |
| wo_bwd_dX | 494 | 758 | 0.65 | 11 |
| qkv_bwd_dX | 819 | 1249 | 0.66 | 11 |
| head_bwd_dX | 1247 | 1546 | 0.81 | 11 |
| dW_qkv | 541 | 1070 | 0.51 | 11 |
| dW_o | 182 | 616 | 0.30 | 11 |
| dW_up | 712 | 1291 | 0.55 | 11 |
| dW_down | 712 | 1392 | 0.51 | 11 |
| dwte | 1170 | 1278 | 0.92 | 14 |
Target >=0.6 met on logits, head dX, MLP up/down, QKV (fwd and dX), dwte. Missed: DGELU epilogue (0.43) and the small weight-gradient GEMMs (0.30-0.55; only 18-72 tiles for 148 SMs, would need split-K). Before this round: 0.12-0.40. Errors vs fp32 reference identical to the mma.sync build (relmax 2.2e-3..3.3e-3 bf16, ~3e-6 fp32).

## 3. Megakernel results (S = 162.4M, 4,000 steps, 8x512 tokens, same data/init)
Loss: step0 11.7106, step19 8.821558, step 3999 **2.157486 -- identical to the mma.sync build** (S) ; F step0 11.8006, final 2.7195 identical to mma F.
| run (B300) | time | ms/step | notes |
|---|---|---|---|
| S tcgen05, split 116/24/8 + live decode (n=2) | **49.9 s / 49.9 s** | 12.46 | decode 1,073-1,085 tok/s |
| S tcgen05, train only, all 148 SMs | 40.1 s | 10.02 | |
| S mma.sync, split 116/24/8 (same day, same GPU) | 90.1 s | 22.52 | decode 1,788 tok/s |
| F tcgen05, split 116/24/8 | **78.2 s** | 19.56 | decode 618 tok/s |
| F tcgen05, train only | 62.7 s | 15.67 | |
Gains by step (S split, ms/step): first working tcgen05 (v1-style) ~14.4 train-only; wide/nsub pipelining + 1 block/SM 11.07 train-only; smem-cached cross-entropy (`-DCE_CACHE=48`) -0.1; staged epilogue + prefetch 12.46 split. Neutral/worse (tried): narrow-only 14.5, narrow for TN 13.0, slot multipliers 0.5/1.5/2/3 (13.5-14.0), early AdamW phases (neutral), 3 wide stages, wide nsub<=2.

## 4. Comparison (all re-measured on the pinned B300, FA's numbers are their B200)
| | ours | baseline | ratio |
|---|---|---|---|
| S split (116 SMs) vs PyTorch eager MPS-116 (warm 55.9 s; cold 63.8 s) | 49.9 s | 55.9 s | **1.12x faster** |
| S train-only full GPU vs eager full (48.6 / 49.0 s) | 40.1 s | 48.8 s | 1.22x faster |
| S split vs Megatron-core local (no TE), MPS-116 (136.6 s; round 1: 124.5 s) | 49.9 s | 136.6 s | 2.7x faster |
| S split vs our mma.sync megakernel | 49.9 s | 90.1 s | 1.8x faster |
| F split vs eager F MPS-116 (79.9 s) | 78.2 s | 79.9 s | 1.02x (parity) |
| F train-only vs eager F full (65.7 s) | 62.7 s | 65.7 s | 1.05x |
| decode S (24 SMs) vs vLLM ~820 tok/s (round-1 number, not re-measured) | ~1,080 tok/s | 820 | 1.3x (FA 1.52x); was 2.1x with mma.sync |
FA's framing (their megakernel 1.52x faster than Megatron+TE; 34% slower than ...) is not reproducible without TE (TE still does not build here). Versus the only strong no-TE baseline, eager PyTorch, we now beat it on S (12%) and tie on F, where in round 1 we were 1.56x/2.4x slower. Honest limits: decode throughput dropped because tcgen05 forces 1 block/SM; F is only at parity; ~39% of step time is non-GEMM (cross-entropy ~1.2 ms, embedding-bwd ~1.2 ms, attention bwd, AdamW tail) still on mma.sync/elementwise code.

## 5. Spend
GPU seconds tracked in `results/r2_spend.jsonl` (rate $0.001736/s): 2,163 s = **$3.75** (probes/tests ~$1.0, debugging ~$0.5, tuning ~$0.9, baselines $1.02 (2 containers, B300, a: 254 s, b: 334 s), final runs ~$0.8) plus a few untracked sub-10 s calls and CPU build containers (each 1-2 min). Far under $25; no failed-round limit hit (the hang took ~12 diagnostic rounds, each progress-making).

## 6. Files (new only)
`gemm_tcgen05/{gemm_sm100_v2.cuh,gemm_backend_r2.cuh,make_test_r2.sh,test_gemm_r2.cu,occ_probe.cu,mbar_probe.cu}`, `megakernel/{ofa_r2.cu,build_r2.sh}`, `modal/app2.py`. Build line used: `MAXREG=255 GENCODE='-gencode arch=compute_100f,code=sm_100f' NODOOM=1 ./build_r2.sh sm_100f out -DOFA_GEMM_BACKEND=2 -DG2_FORCE_TC -DG2_PERSIST_TMEM -DOFA_BLOCKS_PER_SM=1 -DCE_CACHE=48` (F: `MODEL=F`). Doom stayed off (stubs). `mbar_probe` showed generic `ld.shared` of an mbarrier word returns 0/stale, so barrier-state dumps from device code are unreliable.

## 7. Open
Split-K for small dW GEMMs, DGELU epilogue, tcgen05 attention, fused CE, recovering decode speed (2 blocks/SM impossible with TMEM in the same kernel), vLLM/TE re-baselining, `modal app list` at the end: all ofa apps `stopped`, 0 tasks.
