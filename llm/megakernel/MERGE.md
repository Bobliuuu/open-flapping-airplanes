# MERGE.md: what to port from `llm/megakernel/triad_llm.cu` into the Doom/dashboard copy (`llm/triad/`)

Everything below is already in `llm/megakernel/triad_llm.cu`. `triad_llm_perf.patch` is the complete diff against
`triad_llm.cu.orig` (the file as first integrated, plus `#include <unistd.h>`), so
`patch -p0 triad.cu < triad_llm_perf.patch` should apply to a copy that only added Doom/dashboard code. Hunks are
independent; port in this order.

## 1. Scheduler claim: CAS loop -> unconditional fetch-add (the big win, ~2x)
* Add `#include <unistd.h>` (needed for `usleep`).
* After `constexpr int MAXPH = 1024;` add `#ifndef CLAIM_FETCHADD / #define CLAIM_FETCHADD 1 / #endif`.
* In `job_loop`, thread 0's claim section: under `#if CLAIM_FETCHADD` take the ticket with
  `atomicAdd(J.ticket, 1u)` first (`g >= total` -> exit), then decode `(step, phase, local)` from `g` and poll the
  phase's dependencies (`ld_acq`, `__nanosleep`). Decode job (`job_id == 1`) keeps its `finished` early-out inside the wait.
  The original CAS loop stays under `#else`.
* Why: one CAS retry loop on one ticket word serialises at about 1.9 claims/us. A step has 125k-127k tasks, so claiming
  alone cost about 66 ms/step. Fetch-add cannot deadlock because every dependency points to a lower ticket.
* Dependencies must stay "lower ticket" only: phases are created in topological order and cross-step deps point to the previous step.
* Doom's own task queues (priority Doom > infer > train) are separate and unchanged. If the Doom fork routes its tasks
  through the same `job_loop`, give them the same fetch-add treatment.

## 2. AdamW overlap (about 1 ms/step on S, hides the whole optimizer tail on F)
* `adam_hp(C, step)` device helper (after `lr_at`).
* `K_EMB_WTE` case: after `embed_bwd_wte_task`, if `d.i0` run `adamw_task<true>` over this task's 128 vocab rows
  (`RN = EMB_VCHUNK*D` elements at `local*RN`). The empty nodecay list MUST be passed as `(const int2*)d.p[7], d.i1` (run-time
  values); passing `nullptr, 0` lets the compiler pick a different FMA pattern and results are no longer bit-identical to the stand-alone AdamW.
* Host graph: new `--opt-mode` flag (0 = single AdamW phase after the step, as before; 1 = per-layer groups; 2 = default, groups + wte fused into
  embed-bwd). Per-layer `group()` lambda (relative `nodecay` list per group, layer L-1 also covers `gf`), `layer_grads[l]` /
  `layer_g[l]` bookkeeping in the backward loop, `pv(n)` placeholder cross-step deps (`make_int2(-1,-1)`, patched after the groups exist),
  the 1-task `"step barrier"` phase that ends the step (`last_ph`), and the patch block just before `const int nph`.
* Cross-step dependencies now are: `data` and `embed` wait for the previous step's wte phase, `embed` also for the wpe AdamW, and layer l's first
  forward phase (`L%d ln1`) waits for layer l's AdamW. The decode job is untouched: it reads the live bf16 weights as before.
* Check: `--opt-mode 0/1/2` give bit-identical losses (verified on S, 30 steps).

## 3. Config F (`-DMODEL_F`: L=4, d=2048, H=32, MLP 8192, 408M params)
* `#include "decode_wide.cuh"` instead of `"../decode/decode.cuh"`. `decode_wide.cuh` is `llm/decode/decode.cuh` with the
  `D_ <= 1024` assert relaxed to 2048 and `ModelF = Cfg<4,2048,32,512,100352>` added. It passes decode's own accuracy test for F
  (`decode_f/`, max rel err 6.5e-3, argmax agreement 100%).
* Model constants block: `#if defined(MODEL_F) constexpr int L = 4, D = 2048, H = 32; using DC = decode::ModelF;` before the `MODEL_M` branch.
* Build: `./sync_build.sh <bin> -DMODEL_F`; data: `MODEL=F python3 make_init_flat.py` -> `~/data/init_flat_F.f32` (from `llm/baseline/golden/F`).
* Nothing else is shape specific: GEMM/attention/LN/CE/AdamW libraries already take the sizes as template parameters.
* Smem/regs: F kernel has 1464 B stack and about 1.9 KB spills (S: 560 B/732 B) but the same 216 blocks x 64 KB; no `static_assert` changes were needed.

## 4. Files that are only tooling (copy if useful)
`sync_build.sh`, `runS.sh`, `runF.sh`, `make_init_flat.py` (now `MODEL=S|F`), `check_grads.py` (now `MODEL=S|F`), `ref_F.py` + `compare_F.py` (PyTorch reference and overlay),
`bundle1.sh`/`bundle2.sh` (examples of one-lock-hold validation/timing runs), `decode_f/` (decode F test).

## Do NOT port
`triad_llm.cu.orig` and `triad_llm_perf.patch` are bookkeeping. No library header (`gemm`, `attention`, `elementwise`, `decode`) was modified; `decode_wide.cuh` is a copy.
