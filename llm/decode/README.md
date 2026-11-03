# decode: batch-1 autoregressive decode of model "S" as megakernel tasks

Header-only `decode.cuh` (namespace `decode`), test `test_decode.cu`, reference `ref_decode.py`, driver `run_test.sh`.
Build: `nvcc -arch=sm_80 -O3 -std=c++17 test_decode.cu -o decode_test`. Reads the SPEC bf16 weights (W[in,out] row-major, wte[V,d],
wpe[T,d], gamma-only LN) directly, so it can serve from the bf16 working copy the optimizer refreshes.

## Contract
* Every task is a `__device__ __forceinline__` function that all 256 threads of a block call together; no `blockIdx`, no grid sync,
  no float atomics. Dynamic smem <= 11.9 KB (`smem_bytes<C>()`, far below the 80 KB budget; 128 B aligned base). Any number of
  blocks, any order inside a phase. Registers: <= 102 per task (fits 2 blocks/SM).
* Single entry point: `run_task<C>(const Params<C>& P, int phase, int task, char* smem)`; `phase_tasks<C>(phase)` gives the task count.
  `prefetch_task<C>(P, phase, task)` (optional) pulls that task's weights into L2.
* Phase p+1 may start only after ALL tasks of phase p finished (the caller's dependency counter). Phase 0 of step s+1 follows the
  SAMPLE phase of step s. `C = ModelS` (L=12 d=768 H=12 T=512 V=100352); `ModelM` (L=16, d=1024, H=16) compiles too (not verified).
* `Params<C>`: `Weights w` (pointer per layer, 16 B aligned), `State<C>* st` (device memory from `alloc_state<C>(out_cap)`),
  `temperature` (<=0 greedy), `seed`, `valid_vocab` (tokens >= it are never sampled, e.g. 100277), `forced/nforced` (teacher forcing:
  the token fed at step s+1 is `forced[s+1]` while s+1 < nforced; this is also how a prompt is prefilled), `auto_reset`.
* Host helpers: `alloc_state`, `begin_context(P, token, pos=0)` (sets token/pos/step and embeds the token), `launch_step_grid(P, stream)`
  (one kernel launch per phase, grid = task count), `k_persistent` (the persistent reference scheduler, see below).
* State: token, pos (0..T-1), step, `kv` bf16 `[L][2][H][T][64]` (index `((((l*2+kv)*H+h)*T+pos)*64+i)`, k=0 / v=1), `logits` fp32 `[V]`,
  `out_tokens[step]` (the sampled token, independent of teacher forcing). The SAMPLE task advances `pos`, `step` and writes the
  embedding of the next token. **Context-full policy:** after position T-1 the task sets `ctx_full=1`; with `auto_reset=1` the context
  restarts at pos 0 with the sampled token as only context (simplest policy); with `auto_reset=0` the caller must stop or call
  `begin_context`. No sliding window / re-prefill is implemented. The KV cache never needs clearing (rows > pos are never read).

## Phases per decode step (S): 3 per layer + head + sample = 38
| phase | tasks | granularity / what a task does | weight bytes read |
|---|---|---|---|
| QKV (x12) | 36 | residual read + LN1 (every task, 768 values) + GEMV for 64 output columns of [q\|k\|v] (768 rows x 128 B); k,v are appended to the KV cache, q to `st->q` | 98 KB / task |
| ATT+WO (x12) | 36 | (head h = t/3, 256-column block cb = t%3): softmax attention of head h over pos+1 keys (recomputed by the 3 tasks of a head, KV is L2 resident), then o[64] . Wo[64h..+64, 256cb..+256]; result added to the residual | 33 KB + KV (<=128 KB) |
| UP+DOWN (x12) | 48 | residual read + LN2 + GEMV for 64 hidden units, GELU(tanh), bf16-round, then h[64] . Wdn[64t..+64, :]; result added to the residual | 98 KB + 98 KB |
| HEAD | 784 (HR=128) | LN_f + 128 vocab rows of wte (1536 B contiguous per row; warp per row group, shuffle reduction); writes logits fp32 and chunk-local argmax / Gumbel-max candidates | 196 KB |
| SAMPLE | 1 | deterministic reduction of the 784 candidates (ties -> lowest id), teacher forcing, pos/step update, embeds the next token | - |

Weight bytes per token: 12 x (768x2304 + 768x768 + 2x768x3072) x 2 B + 100352x768x2 B = 170.1 MB + 154.1 MB = **324.2 MB**.

Design points. The residual stream is kept in **fixed point** (int64, 2^-30 units) and updated with 64-bit integer atomics by the
ATT+WO and UP+DOWN tasks: integer addition is associative, so results are bitwise independent of task order/placement without any
partial-sum buffers (two copies, `xw` master / `xr` read by UP, see the comment in `State`). All other reductions have a fixed order.
GEMV tasks issue the first batch of weight loads (16 B, `ld.global.L1::no_allocate`, coalesced along `out`) before the prologue so DRAM
latency overlaps the LayerNorm. Weights/KV are never read through the non-coherent path, so refreshed weights are always seen
(visibility of an optimizer refresh is the caller's phase ordering).

Sampling: greedy = argmax of logits; temperature T>0 = Gumbel-max over `logit/T + g(seed, step, vocab id)` with
`g` from a counter-based splitmix64 hash; chunk-local best + a fixed-order final reduction => identical stream for every schedule.

## Persistent scheduler (reference, in `k_persistent`)
Blocks on SMs `< sm_limit` (at most `per_sm` per SM) take part (so a partition of the GPU can be emulated; the rest exit). Task of a
phase: block rank r takes task r first (no atomic), further tasks (only for phases with more tasks than blocks, i.e. HEAD) come from an
atomic ticket, one claimed ahead. After its tasks a block does `red.release.gpu` on the phase counter and polls with `ld.acquire.gpu`
(exponential `nanosleep` backoff). `scramble=1` permutes the task order per phase and inserts random delays. `nsteps` decode steps run
inside ONE launch (the sampled token feeds back on the device). `prefetch` bit 0: L2-prefetch the claimed-ahead task; bit 1: while waiting
at a barrier prefetch the next phase's tasks. Neither helped measurably (see below).

## Measured (A100-SXM4-40GB, shared box; numbers only from runs where the GPU was idle at start, noise ~ +-5%)
Weight bytes per token 324 MB (170 MB layers + 154 MB tied head). tokens/s and effective GB/s = 324 MB / time per token. Steady-state,
greedy, 128 consecutive steps from position 0 (context 0..127) and from 384 (context 384..511, more KV traffic).

| configuration | pos 0..127 | pos 384..511 |
|---|---|---|
| plain grid, 38 launches/step, CUDA graph | 549 us, **1821 tok/s, 590 GB/s** | 596 us, 1679 tok/s, 544 GB/s |
| persistent, 108 SMs, 1 block/SM | 530 us, **1887 tok/s, 611 GB/s** (39% of 1555 GB/s peak, 45% of the 1350 GB/s streaming ceiling) | 577 us, 1732 tok/s, 561 GB/s |
| persistent, 108 SMs, 2 blocks/SM | 596 us, 1677 tok/s, 543 GB/s | 682 us, 1467 tok/s, 475 GB/s |
| persistent, 34 SMs x 2 blocks | 633 us, 1581 tok/s, 512 GB/s | 685 us, 1460 tok/s, 473 GB/s |
| persistent, **17 SMs x 2 blocks** | 955 us, **1047 tok/s, 339 GB/s** | 1038 us, 963 tok/s, 312 GB/s |
| persistent, 17 SMs x 1 block | 1279 us, 782 tok/s, 253 GB/s | 1395 us, 717 tok/s, 232 GB/s |

(The 24-SM row of the published demo, ~1076 tok/s on 24 of 148 SMs, compares to 24 SMs x 2 blocks here: 678 us = 1474 tok/s; the
17-SM partition reaches 1047 tok/s, i.e. about the demo's number.) A pure streaming kernel (16 B loads, no dependencies) on this
box gets 1348 GB/s with >= 108 blocks (87% of peak), 710 GB/s with 34 blocks on 34 SMs, 372 GB/s with 17 blocks: per-SM streaming
bandwidth is only ~21 GB/s per block, so ~17 SMs cannot exceed ~0.4-0.7 TB/s in any case.
The L2-prefetch options (`prefetch` bit 0/1) were neutral to slightly negative. GPU sharing: other jobs on the GPU raised some runs by 2x;
the table only contains runs where `nvidia-smi` showed 0% at the start (later parts of some runs were still contaminated and discarded); repeats of the same configuration differ by ~5%.

Per-phase timeline (108 SMs x 1 block, from the `globaltimer` stamps, time between phase completions): QKV 8.5 us, ATT+WO 9.9 us,
UP+DOWN 16.2 us (sum 34.6 us/layer, x12 = 415 us), HEAD 145 us (1065 GB/s, 79% of the streaming ceiling), SAMPLE 5 us -> 565 us.

## Accuracy
`./run_test.sh` (weights random bf16: embeddings N(0,0.02), layer matrices N(0,0.025) (Wo/Wdn 4/sqrt(2L) smaller), gamma = 1 + 0.1 N(0,1),
so that every layer changes the logits; 32 teacher-forced steps through the decode path, compared with a full-sequence fp32 PyTorch
forward of the same bf16 weights): **max rel. logit error (max|d|/max|ref| per step) = 5.3e-3 (mean over steps 4.6e-3), argmax agreement 100% (32/32,
top-1/top-2 margin median 0.077, min 0.004 in logit units), PASS** (limits: 3e-2 and 99%).
Determinism test (96 steps: 8 forced prompt tokens then temperature-0.9 Gumbel sampling): the plain-grid result and persistent runs with 108x2,
108x2 scrambled order + random delays, 17x2 scrambled, 17x1 + prefetch, 5x1 scrambled and a single block give **bitwise identical tokens,
final logits and the whole 18.9 MB KV cache**; a different seed gives a different stream. PASS.

## What limits speed
Bandwidth is not the limit when all SMs are used; latency is. The big matrices are small per layer (QKV 3.5 MB, Wo 1.2 MB,
up+down 9.4 MB: 3-7 us of HBM time each) and each phase is a chain of dependent round trips: residual read -> LN (block reductions) -> weight
loads -> FMAs -> cross-row reduction -> atomics -> release/acquire barrier -> next phase. One layer = 3 phases x ~10 us ~= 35 us versus ~10.5 us of
pure HBM time, so the 12 layers take ~415 us of the 530 us while the LM head (154 MB, 784 independent tasks) streams at 1065 GB/s. Therefore
the >= 70% of peak HBM goal is NOT reached for the full step (39% of 1555 GB/s; the head alone is 68% of peak / 79% of the practical 1348 GB/s ceiling);
the layers run at ~30% (and the persistent barrier costs ~1 us/phase, the plain-grid CUDA graph is within 4%). What was done about it:
LN fused into the GEMV prologue, residual merge via integer atomics instead of partial-sum buffers, attention fused with Wo and up with
down (5 -> 3 phases/layer, took layer time from ~50 to ~35 us), first weight batch loaded before the LN, tasks sized for 36-48 per phase.
At 17 SMs the step is closer to bandwidth/per-SM limited (timeline with 17 SMs x 1 block: head 460 us = 335 GB/s, layer ~75 us; 17 SMs x 2 blocks: 955 us/token in the clean run).
Not tried: split-K for the QKV GEMV to get > 36 tasks, 128-bit `cp.async` multi-stage tiles through smem (needs 96 KB per QKV task),
cluster/DSMEM fusion (Hopper+), keeping LN statistics out of every task by an extra phase, FP8 weights. 2 blocks/SM is slower than 1 block/SM
at 108 SMs in this latency-bound regime (more co-resident blocks contend for the same SM issue/LSU slots without more useful bytes in flight; the
caller can pick 1 block per SM for decode-only phases). KV traffic is small (<= 19 MB per token at pos 511: ~4% of weight bytes) but adds ~3 us/layer
of dependent loads at long context (596 vs 549 us).
Other limits: S only verified; ModelM compiles but is untested; the persistent loop's spill (120 B local) comes from the scheduler variables,
tasks themselves have 0 spills; vocab `V % HR == 0`, `D % 256 == 0`, `D <= 1024`; `temperature` sampling has no top-k/top-p; padded vocab ids
>= `valid_vocab` are excluded from sampling but still written to `logits`; the fixed-point residual saturates at |x| ~ 8.6e9 (never in practice);
KV cache rows are overwritten in place when `auto_reset` wraps.

## B200 notes (description only, nothing implemented)
The same decomposition applies, but the weight stream is ~5x faster (8 TB/s HBM3e) while per-phase latency (barrier + dependent L2 round
trips, ~5-10 us/phase here) does not shrink, so a B200 would be even more latency bound at batch 1: 38 phases x ~10 us would cap it
near 2.5k tok/s unless phases are fused harder. Levers there: thread block clusters with DSMEM to keep LN + QKV + attention of one head inside a cluster
(removing global barriers), TMA bulk loads / multicast for the weight tiles (deeper in-flight bytes per SM than 16 B loads),
tcgen05/TMEM for the GEMVs is pointless at batch 1 (CUDA cores / `fma.f32x2` suffice), fp8/fp4 weights (the demo's 2x-4x lever) and the
148 SMs with 126 MB L2 would hold ~3/4 of a layer-stack's working set, i.e. cross-token L2 residency of the early layers.
