# B300 experiment report (Modal, `gpu="B200+"` landed on an NVIDIA **B300 SXM6**, cc 10.3, 148 SMs)

**Everything below is B300, not B200.** Driver 610.57.04 / CUDA UMD 13.3, container CUDA toolkit 13.0 (nvcc 13.0.48), torch 2.13.0+cu130, vLLM 0.32.0, megatron-core 0.16.1.
Billed at the B200 rate ($0.001736 per GPU-second). FA's targets (Megatron 70.1 s, 230k tok/s, vLLM 710 tok/s, KVM 46.1 s) are their B200 numbers read from the video; they are not reproduced here.

**Transformer Engine could not be built** (sdist `transformer_engine_torch` 2.19/2.20 fails in `csrc/extensions/ep.cpp`: `ncclWindow_t` vs `ncclWindow_vidmem*` against the image's NCCL; 4 attempts), and flash-attn was not tried. **All baselines are therefore no-TE**: PyTorch eager with `F.scaled_dot_product_attention`, and Megatron-core with the *local* layer spec (unfused attention). They are weaker than FA's Megatron+TE; the PyTorch-eager number is the strongest baseline we have.

## 1. Device (probe)
B300 SXM6 AC, 148 SMs, 232,448 B opt-in smem/block (233,472 B/SM), 126 MB L2, 287 GB HBM, max SM clock 2032 MHz, copy bandwidth 6.4 TB/s (read+write, 42.9 GB moved), `%smid` 0..147 contiguous.
Binaries that ran: `sm_100a`, `sm_100f`, `sm_103a` (tiny probe), `sm_103f`, `compute_100` PTX (JIT), `sm_80`, `sm_90a`; `compute_100a` PTX-only did **not** run (no kernel image). Later an `sm_103a` build of the GEMM test failed with "no kernel image is available" (unexplained; sm_100a SASS works).
MPS works (root, user-level daemon). SM counts under MPS: 8%→10, 10%→14, 12→16, 14→20, **16→22**, 18→26, 20→28, 25→36, 50→74, 70→102, 74→108, 76→112, 78→114, **79→116**, 80→118, 82→120, 85→124, 90→132, 100→148. 116 SMs = 79%. A 24-SM share would be ~17% (22 SMs at 16%, 26 at 18%); **17% was used for vLLM but its SM count was not measured**.

## 2. Baselines (4,000 steps, batch 8x512 tokens, each job alone on the GPU; "MPS-116" = 79%)
| job (model S, 162.4M params) | n | time | tok/s | final loss |
|---|---|---|---|---|
| PyTorch eager, full GPU | 2 | 71.2 s (cold) / **48.1 s** (warm) | 230k / **340.6k** | 2.1447 / 2.1433 |
| PyTorch eager, MPS-116 SMs | 2 | **55.9 s** / 55.9 s | 293.1k | 2.1589 / 2.1494 |
| PyTorch eager + whole-step CUDA graph, full | 1 | 55.5 s | 295.2k | 2.1429 |
| Megatron-core local spec (no TE), full | 1 | 132.7 s | 123.5k | 2.1398 |
| Megatron-core local spec (no TE), MPS-116 | 1 | 124.5 s | 131.6k | 2.1398 |
| vLLM decode batch 1, 256 tok, full GPU | 2 | – | 807, 818 | – |
| vLLM decode batch 1, MPS 17% (SM count unverified) | 2 | – | 825, 818 | – |
| PyTorch eager, **model F** (407.9M), full | 1 | 71.5 s | 229.1k | 2.6884 |
| PyTorch eager, model F, MPS-116 | 1 | 80.4 s | 203.7k | 2.6884 |
Notes: run 1 of PyTorch eager was cold (PTX JIT cache empty, 71 s vs 48 s warm); the first Megatron run was probably cold too (full slower than MPS-restricted is not physical). The GPU had one compute process (two with the MPS daemon) in every run (`max_compute_procs_seen`). CUDA-graph being slower than eager and the Megatron full/MPS inversion are unexplained. Raw JSON: `bl_g1/`, `bl_gF/`, `bl_g1_summary.json`, `bl_gF_summary.json`.

## 3. Our megakernel on B300 (mma.sync backend, `sm_100a` build run on cc 10.3; Doom OFF: stubs, its 8 SMs idle)
Step 0 loss 11.7106 and step 19 loss 8.821558 match the A100 build (11.710649 / 8.821497). Data: same 73.9M-token TinyStories cl100k set and sampler.
| run | split | 4,000 steps | ms/step | tok/s | decode tok/s | final loss | SM-equiv busy |
|---|---|---|---|---|---|---|---|
| S, train+decode | 116/24/8 (8 idle) | **87.1 s** | 21.77 | 188.2k | **1,762** | 2.1575 | 103.2 train / 17.7 serve |
| S, train only | all 148 | 72.7 s | 18.17 | 225.4k | – | 2.1575 | 123.2 |
| S, GEMM_STAGES=6 (600 steps) | 116/24/8 | (89.1 s proj.) | 22.27 | 183.9k | 1,743 | – | – |
| S, native sm_103a build (600 steps) | 116/24/8 | (89.9 s proj.) | 22.49 | 182.2k | 1,781 | – | – |
| **F**, train+decode | 116/24/8 (8 idle) | **189.9 s** | 47.48 | 86.3k | **815** | 2.7195 | 108.4 / 20.6 |
F uses the SPEC init (seed 1234) rather than the A100's golden F init; its step-0 loss 11.8006 equals the A100 F value. 

### Comparison (same 116/24 SM share, no-TE baselines)
| | ours | baseline | ratio |
|---|---|---|---|
| S train vs Megatron-core local (no TE), MPS-116 | 87.1 s | 124.5 s (n=1) | ours 1.43x faster |
| S train vs PyTorch eager, MPS-116 | 87.1 s | 55.9 s | ours **1.56x slower** (56% slower) |
| S decode vs vLLM (24-SM share, 17% MPS) | 1,762 tok/s | ~820 tok/s | **2.1x** (FA: 1,076 vs 710 = 1.52x) |
| F train vs PyTorch eager, MPS-116 | 189.9 s | 80.4 s (n=1) | ours **2.4x slower** |
FA's claim (34% slower than a fused Megatron+TE) is **not reproduced and not testable here**: without TE the only strong baseline is eager PyTorch, which beats our kernel on B300 because cuBLAS reaches 1.0-1.7 PFLOP/s while our `mma.sync` GEMM reaches 0.1-0.4 PFLOP/s (Section 4). On the A100 the same kernel beat eager; on Blackwell the missing tensor-memory GEMM dominates.
Launch counts were not measured on B300 (the kit's count phases were not run).

## 4. GEMM: mma.sync control vs real tcgen05 (stand-alone, model-S shapes, vs cuBLAS)
**Correction to the first test log (`gemm_test.log`, "OVERALL PASS on all 15"):** those runs did **not** execute tcgen05. The test binaries embed generic `compute_100` PTX (no tcgen05) next to the `sm_100a` SASS; with PTX JIT allowed the driver JIT-compiled the generic PTX, i.e. the mma.sync fallback ran (same registers 128/126, same TF within 0.5%, same errors as the control `t_mma`). With `CUDA_DISABLE_PTX_JIT=1` and one process per case, the SASS runs: registers drop to 48-108 and the tcgen05 path is exercised. All 15 cases then **PASS** against the fp32 reference (relmax 1.6e-6 to 3.3e-3, identical to the control) — so the single-CTA TMA + `tcgen05.mma` + TMEM backend is **numerically correct on this B300** — but a whole-suite run (`t_nt`, mask 2) aborts with "PTX JIT compilation was disabled" because its fallback instantiations only exist as PTX; the persistent-vs-grid bit-exactness / RES-alias columns were reported as in the test (ok where applicable).
| case | tcgen05 TF | mma.sync TF | cuBLAS TF | tc/cuBLAS | tc/mma.sync | regs/thr | check |
|---|---|---|---|---|---|---|---|
| qkv_fwd | 253 | 336 | 1192 | 0.21 | 0.75x | 71 | PASS |
| wo_fwd | 108 | 191 | 665 | 0.16 | 0.57x | 48 | PASS |
| up_fwd | 151 | 257 | 1227 | 0.12 | 0.59x | 72 | PASS |
| down_fwd | 293 | 253 | 1228 | 0.24 | 1.16x | 48 | PASS |
| logits | 252 | 376 | 1197 | 0.21 | 0.67x | 72 | PASS |
| down_bwd_dX | 149 | 269 | 1226 | 0.12 | 0.55x | 53 | PASS |
| up_bwd_dX | 294 | 249 | 1236 | 0.24 | 1.18x | 48 | PASS |
| wo_bwd_dX | 170 | 211 | 635 | 0.27 | 0.81x | 71 | PASS |
| qkv_bwd_dX | 247 | 240 | 1122 | 0.22 | 1.03x | 48 | PASS |
| head_bwd_dX | 558 | 279 | 1520 | 0.37 | 2.00x | 71 | PASS |
| dW_qkv | 389 | 241 | 1000 | 0.39 | 1.61x | 72 | PASS |
| dW_o | 134 | 81 | 558 | 0.24 | 1.66x | 72 | PASS |
| dW_up | 515 | 320 | 1276 | 0.40 | 1.61x | 72 | PASS |
| dW_down | 506 | 320 | 1281 | 0.40 | 1.58x | 72 | PASS |
| dwte | 426 | 414 | 1478 | 0.29 | 1.03x | 72 | PASS |

Speed: **0.12-0.40 of cuBLAS**, same range as mma.sync (0.14-0.37); tcgen05 is 1.6-2.0x faster than mma.sync on the K=4096/100352 weight-gradient and head GEMMs (head_bwd_dX 558 TF, dW_up 515 TF), and 1.2-1.8x *slower* on the K=768 GEMMs (qkv, wo, up, logits, down_bwd_dX). What limits it (from the code, not profiled): single CTA 128x128 (no 2-CTA, no TMA multicast), a 3-stage pipeline with no persistent scheduler so the TMEM accumulator is not double-buffered (MMA idles during the epilogue), TMEM allocated/freed per tile, and the epilogue stores uncoalesced 4-8 B per thread; short-K cases are epilogue/latency-bound. cuBLAS gets its 1.0-1.7 PF from 2-CTA 256x128 tiles and overlapped epilogues.
**Megakernel with the tcgen05 backend: does not work yet.** `ofa_tc_100a` (S, 20 steps) printed no loss and was killed after 90 s in every attempt (JIT allowed and forbidden; once "PTX JIT compilation was disabled" at ofa.cu:431); the `sm_103a` build fails with "no kernel image"/"named symbol not found". Likely the mbarrier/proxy-fence risks (R3/R4 in `gemm_tcgen05/README.md`) inside the persistent scheduler; not debugged. Quick to *try* (flag `-DOFA_GEMM_BACKEND=1`, already built), not quick to fix.

## 5. Spend (GPU-seconds x $0.001736; wall time of each GPU function body, container start/idle excluded)
| function | GPU-s | $ |
|---|---|---|
| probe | 39.5 | 0.069 |
| megakernel S runs (loss20, split, train-only, mma6, sm_103a) | 213.9 | 0.371 |
| baselines S (eager x4, graph, Megatron-local x2, vLLM x4) | 970.2 | 1.684 |
| GEMM test round 1 (mma.sync fallback, mislabelled tcgen05) | 27.9 | 0.048 |
| megakernel F + sm_103a tcgen05 attempts | 224.3 | 0.389 |
| baselines F (eager x2) | 181.1 | 0.314 |
| run3 (aborted tcgen05 attempts) | 5.7 | 0.010 |
| run4 (per-case tcgen05, megakernel tc timeouts) | 530.7 | 0.921 |
| **GPU total** | **2,193 s** | **$3.81** |
Plus: an earlier B200+ smoke test (about $0.05, not measured here) and Modal **CPU** time for image/data/compile builds (cpu=16 for the vLLM/Megatron/TE image builds and 4 TE attempts, cpu=8 for compiles, 16 for TinyStories tokenisation): estimated roughly $1-2, **not metered**. All under the $40 cap.

## 6. Nothing is running
`modal app list` after the last run: every `ofa-b300` app is `stopped` with 0 tasks (see end of file for the final listing). The only other entry, `okc-dev-h200`, is not ours.

## 7. Unverified / open
* No TE/flash-attn baseline, so FA's "34% slower than Megatron(TE)" is not testable; B300 is not B200.
* n=1 for Megatron-local and F baselines; first runs were cold-JIT.
* 17% MPS ~ 24 SMs not measured; vLLM is CPU/launch-bound at this size, so it barely depends on the share.
* Doom was off (stubs): the 8 Doom SMs were reserved but idle; no Doom on Blackwell tested.
* sm_103a cubin failure and the tcgen05 megakernel hang are not diagnosed.
* Raw outputs in this directory: `probe.json`, `mk_*.txt`, `mk_results.json`, `r2_*`, `r3_*`, `r4_results.json`, `gemm_*.txt`, `bl_g1/`, `bl_gF/`, logs `*.log`; code in `../modal/app.py`.

### Final \`modal app list\` (head)
```
                                      Apps                                      
                                                                                
                              Descript                     Created    Stopped   
  App ID                      ion        State     Tasks   at         at        
 ────────────────────────────────────────────────────────────────────────────── 
  ap-lFwLDZxPmtvjL5EEWEFmfd   ofa-b3   stopped   0       2026-11-   2026-11-  
                              00                           01 23:44   01 23:54  
                                                           EST        EST       
  ap-1dv8zzgbfOZfqYHds4W9km   ofa-b3   stopped   0       2026-11-   2026-11-  
                              00                           01 23:43   01 23:43  
                                                           EST        EST       
  ap-F5sMuFz7EJ7p8K6h9EL0E3   ofa-b3   stopped   0       2026-11-   2026-11-  
                              00                           01 23:38   01 23:42  
                                                           EST        EST       
  ap-ZLcEPgwcFanBkDif3L3Yq9   ofa-b3   stopped   0       2026-11-   2026-11-  
                              00                           01 23:38   01 23:42  
                                                           EST        EST       
  ap-YcWTiMskkY04n4dByGQ7RK   ofa-b3   stopped   0       2026-11-   2026-11-  
                              00                           01 23:37   01 23:38  
                                                           EST        EST       
  ap-mFTQCyLGBanAHYbeYOr61l   ofa-b3   stopped   0       2026-11-   2026-11-  
                              00                           01 23:37   01 23:38  
                                                           EST        EST       
  ap-StfE5IMCKrDHfVsgOJ51KB   ofa-b3   stopped   0       2026-11-   2026-11-  
                              00                           01 23:36   01 23:37  
                                                           EST        EST       
  ap-7fhuTPrZ8IYGDTeQtal8bd   okc-dev-   stopped   0       2026-11-   2026-11-  
                              h200                         01 23:35   01 23:49  
                                                           EST        EST       
                                                                                
```
