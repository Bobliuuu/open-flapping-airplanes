# Research log: can a persistent megakernel own the GPU? (flapping-airplanes)

**Can we build what Flapping Airplanes says it built: a persistent "megakernel" runtime that owns the GPU's execution layer?**

Short answer: **yes for the serving half, probably for the rest.** This repo is the feasibility case and the build ladder.

> Historical research log, written before the demo existed (sections 1-9 are the original feasibility analysis; 10-12 record what was then built). See the [root README](../README.md) for the finished project. Original status note: research only. Dollar and percent figures are one analyst's pricing from existing measurements and are **provisional** until a second person recomputes them.

---

## TL;DR

| | |
|---|---|
| **Who** | Flapping Airplanes: data-efficiency AI lab. CEO Ben Spector is first author of ThunderKittens and co-first author of Hazy Research's megakernels. $1B Series A at about $6B (Oct 21 2026). |
| **Claim** | "The world's most complete megakernel framework": training, inference and a video game in **one persistent kernel**, scheduled by a "Kittens VM (KVM)". |
| **Public evidence** | **None.** No code, paper or benchmark as of Oct 31. Only the X post and press. Their KVM name traces to a public ancestor: ThunderKittens branch `bvm-unity-pipeline` (May 2025). |
| **Feasibility** | Eight hardware requirements force the design. Public code already implements most of it (Hazy, Cohere, Cursor MoK, Lucebox, MPK). |
| **What stays theirs** | A deterministic training ISA, a multi-job preemptive scheduler, a model-to-instruction compiler. **None is needed to serve tokens.** |
| **Where it pays** | Big on B300 8-GPU decode (about 25 to 30% of step reachable). Small on capped, prefill-heavy RTX stacks (about 2 to 3%). |

**Honest scope.** "Exactly what they do" cannot be verified; they have published nothing. The target is therefore their *claimed capability*, reconstructed from first principles and from the public code of the same people.

---

## 1. Why a data-efficiency lab builds a GPU scheduler

Our reading, not their statement.

```
learn more from each sample
  -> more compute per sample, in small, dependent, mixed steps
     (per-sample updates, test-time learning, an environment in the loop)
        |
one kernel per op runs those steps far below the roofline:
  - every kernel boundary pays launch (1.3-2.1 us), pipeline fill/tail, an HBM round trip
  - a decode op lasts only 5-50 us, so that overhead is a big share of it
  - measured: vLLM/SGLang reach <= 50% of H100 bandwidth at batch 1; Hazy's megakernel 78%
        |
a resident VM that runs ANY task graph near the limit makes those algorithms cheap to explore
        |
"the GPU as the primary surface for AI research"
(the video game = an environment running beside the learner)
```

## 2. The design, reverse engineered

Each requirement forces one mechanism, and public code makes the same choice.

| Requirement | Forced mechanism | Seen in |
|---|---|---|
| Switch cost far below task cost | One resident block per SM for the whole program | Hazy, Cohere, MoK |
| One binary holds every task type | Warp roles with `setmaxnreg`; one smem layout per opcode | Hazy `config.cuh`; Cohere dropped smem paging ("a steady source of bugs") |
| Dependent ops overlap tile by tile | Dependency counters in global memory, **fence before every signal** | Hazy, Cohere, MoK |
| Hide HBM latency across task boundaries | Two-slot instruction pipeline: task n+1 loads into pages task n frees | Hazy controller |
| Data-dependent work (MoE, ragged KV) | Static per-SM lists + atomic claim queues / work stealing | Hazy (+14% at batch 8192), Cohere, MoK (CLC) |
| Deadline jobs share the GPU with throughput jobs | Preemption at task boundaries, priority queues | **Not public** (the "VM" part) |
| Reproducible training | Fixed reduction order | MoK: bitwise identical under any scheduling |
| No host in the loop | Host writes descriptors ahead into a ring; controller warp fetches ahead | Hazy throughput VM: CPU over 90% idle |

One SM of the reconstructed VM:

```
 host (async) --descriptors--> [instruction ring per SM | claim queues]    [counters c[k]]
                                      | fetched one ahead                    ^ +1 after a fence
 +--------------- SM: one resident block --v--------------------------------+-------+
 | controller warp : decode opcode, map shared memory, set semaphores        |       |
 | loader warp     : wait until c[dep] >= target, then TMA weights+acts in   |       |
 | consumer warps  : tensor-core / ALU work (the only op-specific code)      |       |
 | storer warp     : smem -> HBM or peer GPU, fence, c[out] += 1 ------------+       |
 +----------------------------------------------------------------------------------+
 Scheduler (host, before the step): list scheduling on the tile DAG,
 priority = remaining critical path.
```

## 3. When it pays: one equation

```
one kernel per op :  T  = sum t_i + sum b_i (launch, fill, tail) + I (host gaps)
megakernel        :  T' >= max( critical path, Bytes/BW, FLOP/peak, Energy/H ) + overhead,   H = P_cap - P_idle
gain              =  I + (sum b_i - overhead) + (sum t_i - max(...))
```

**Under a power cap, a megakernel recovers unused power, not unused time.** At the cap an op already takes `E_i/H`, so overlap only buys the watts other ops leave idle.

|  | kernel short (<~25 us) | kernel long |
|---|---|---|
| **below the cap** | **BIG**: launches + overlap both convert (B300 8-GPU decode, drafter phase) | overlap converts by (1 - P/H) |
| **at the cap** | launch and fill savings only | **nothing**: rewrite the kernel instead (capped prefill GEMMs, attention) |

Worth, by stack (provisional):

```
share of the step a megakernel can reach
Qwen T21  (RTX, capped, prefill-heavy)   ██ 1.6-2.3%
Gemma C8  (RTX, capped, prefill-heavy)   ███ ~3% (6% at most)
Kimi K3   decode (B300, 8 GPUs)          ███████████████ ~30%
GLM-5.3   decode (B300)                  ████████████ ~25%
```

Lesson from our own record: a "-56 us" test-bed win became "-8.5 us" once the power cap was measured. **Never count overlap gains without a power column.**

---

## 4. Feasibility verdict, by component

| Component | Verdict | Why |
|---|---|---|
| Persistent kernel + per-SM instruction ring | **Feasible now** | Hazy / Cohere / MoK ship it |
| Dependency-counter dataflow (no grid barriers) | **Feasible now** | Same |
| Decode megakernel, B200/B300 | **Feasible now** | MegaKittens (`mk-v2-llama`), WIP but public |
| Decode megakernel, RTX sm_120 + GDN + NVFP4 | **Feasible, port needed** | Lucebox (Apache-2.0, batch 1). sm_120 has no TMEM; CLC work stealing exists |
| Model-to-instructions compiler | **Feasible, public starting points** | MegaKittens, MPK (OSDI 2026), Event Tensor |
| Bitwise-deterministic training instructions | **Feasible, hard** | MoK shows MoE training at 1.41x, bitwise. Backward + optimizer ISA is the real work |
| Multi-job preemptive scheduler (inference + training + game) | **Plausible, unproven** | No public reference. Preempt at task boundaries; first approximate with engine-level priority tiers |
| Their full unified system | **Unknown** | Nothing published. Treat as stretch goal, not baseline |

**Biggest risk is not the idea, it is synchronization bugs.** Our MK14 kernel read bitwise on 7 of 7 test-bed cases, then diverged on 3 of 66 served requests: q/k written through the generic proxy, released by flag to a TMA (async-proxy) consumer with no `fence.proxy.async`. A bitwise bed proves arithmetic, not synchronization.

---

## 5. The build ladder

Principles: public code first; build where the bound is big; generator over hand-written kernels; assume a sync bug until a check proves otherwise.

| # | Step | Cost | Exit gate |
|---|---|---|---|
| 0 | **Price the bound** for each stack: add a power column (P/H per kernel family) and a decode-only split to the roofline ruler | about 1/2 day, CPU only, plus second-person recompute | Bound >= ~10%: build. <= ~5%: work GEMM energy instead |
| 1 | **B300 decode via MegaKittens**; port Kimi / GLM decode ops as new instruction types | unsized; repo is WIP | Beats CUDA-graph replay on the decode step, then a served screen |
| 2 | **RTX drafter-phase megakernel** from Lucebox's sm_120 NVFP4 DeltaNet kernel + Hazy/Cohere task design (per-dependency counters, weight prefetch across 15 drafter passes, CLC) | hours for census, then one-pass prototype | Bitwise vs served chain **and** >= 78% of bandwidth |
| 3 | **Overlap decode with prefill GEMMs** (their headline idea) | one high-rate power trace | Build only if trace shows spare watts during decode (ceiling up to +11% on Gemma, unmeasured) |
| 4 | **A generator, not kernels**: StableHLO to TIRx, or adopt MPK / MegaKittens / agent search | long | One decoder layer bitwise, then a whole decode step |
| 5 | **CI before any timing counts** (see below) | about 1 day | Required for every megakernel |

### Step 5: CI checks that must exist first

- **Barrier accounting**: each opcode's arrive count equals its waiter's target.
- **CPU replay** of the descriptor stream (Hazy `python_vm.py`; AutoMegaKernel's static deadlock/race validator).
- **Fixed reduction order**, so outputs stay bitwise.
- **Determinism at served timing**: replay the same requests, compare bitwise against controls, report the controls' own repeat rate beside it.

### The "share the GPU" idea, for a token seller

Serve at target speed, fill gaps with preemptible batch work. **Price it first as a discounted batch tier through the engine's priority scheduling, no megakernel.** A megakernel adds in-step filling, which at the cap is bounded by (1 - P/H).

### Don't

- Build a whole-model megakernel first on a capped prefill stack: at most about 2 to 3.5% after a large build.
- Count overlap gains on a test bed without the power column.

---

## 6. Fallbacks (if a step dies)

| If this fails | Do this instead |
|---|---|
| MegaKittens too immature for Kimi/GLM ops | MPK compiler path, or Cohere's open decode megakernel as the reference and port by hand |
| sm_120 lacks TMEM / Hazy kernels do not port | Lucebox base; counters + CLC; accept batch 1 first |
| Bound < 5% on capped stacks | Targeted attention/GEMM kernels. On Gemma these made 36 to 69% of series gains with no megakernel |
| Preemptive multi-job scheduler too hard | Priority tiers at the engine level; megakernel only for single-job decode |
| Determinism cannot be held | Ship inference-only; defer training ISA |

---

## 7. Public code to start from

| Code | What | GPUs | Fit |
|---|---|---|---|
| [HazyResearch/Megakernels](https://github.com/HazyResearch/Megakernels), branch `mk-v2-llama` ("MegaKittens") | PyTorch fx graph to DAG optimizer to dispatcher to one persistent kernel; Llama-1B / 70B decode | B200 (sm100a), B300 (sm103a), CUDA 13+ | B300 decode. WIP |
| HazyResearch/Megakernels, `main` / `throughput` | Llama-1B batch-1 VM; Llama-70B TP throughput VM | H100 | Reference design |
| [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens), branch `bvm-unity-pipeline` | `prototype/vm`, `demos/kvm-runner` (May 2025) | H100 / B200 | Origin of the "KVM" name |
| [cohere-ai/cohere-megakernel](https://github.com/cohere-ai/cohere-megakernel) | Decode megakernel, 16 opcodes, counter barriers | H100 BF16 | RTX FP8/FP4 "soon"; no push since Sep 8 |
| [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub) `optimizations/megakernel` | Fused single-kernel decode, Qwen3.5-0.8B (DeltaNet + attention), NVFP4 path | RTX 3090; sm_120/121a | Closest to RTX PRO 6000 + GDN + NVFP4. Batch 1, grid-wide barriers |
| [mirage-project/mirage](https://github.com/mirage-project/mirage) branch `mpk` | HF model to megakernel compiler; 1.2 to 6.7x lower latency | Hopper; Blackwell demos | Compiler alternative. Active |
| RightNow-AI/AutoMegaKernel | Agent harness with statically checked schedule IR | sm_80/90/120 | Static-check idea; tiny models |
| ForgeMegakernel (arXiv, Sep 2026) | Coding agents + design milestones + mid-state test oracle | n/s | Agent method; 1.21x over SGLang |

Also: Cursor's Mixture-of-Kittens (MoE training megakernel, 1.41x, bitwise); CUTLASS SM120 example 87a (CLC tile scheduler on sm_120).

---

## 8. Open questions (each is cheap to settle)

1. **Flapping's code or numbers**: none as of Oct 31. Recheck before citing.
2. **P/H on RTX PRO 6000** for verify attention and decode GEMMs: one `bed_energy.py` run each.
3. **Qwen T21 host idle** reads 2.0 to 2.8 ms in nsys, maybe <= 2 ms without nsys inflation: in-boot A/B.
4. **MegaKittens on Kimi/GLM ops**: untested. RTX needs its own port (no TMEM on sm_120).
5. **Lucebox at batch > 1 and with MTP drafting**: unmeasured.

## 9. Their claims vs. what is public

| Claim | Source | Status |
|---|---|---|
| Models could be "orders of magnitude more data-efficient" | Index (Jan 28), TechCrunch (Feb 10) | thesis |
| "World's most complete megakernel framework" | their X post, Oct 21 | **claim only** |
| "Kittens Virtual Machine" with its own instruction-based programming model | third-party quote-post, Oct 21 | third-party, no numbers |
| "Writing its own version of the software that runs Nvidia's graphics chips" | Bloomberg via TNW | statement |
| $180M seed at $1.5B; $1B Series A at about $6B, NVIDIA participating | TNW, TechCrunch | reported |

## Key sources

Hazy Research: *No Bubbles*, *We Bought the Whole GPU* · Cursor: *Mixture of Kittens* · Cohere megakernels · NVIDIA: *Megakernels Considered Harmful* (2013) · Latent Space: *megakernels are so dead and so back* · MPK (OSDI 2026) · Event Tensor · ForgeMegakernel · AutoMegaKernel.

---

## 10. M0 results: sm_80 (A100-SXM4-40GB, Lambda)

Code: [`m0-a100/mk.cu`](m0-a100/mk.cu) (v2, split-K chunked deps), [`m0-a100/mk_v1.cu`](m0-a100/mk_v1.cu) (v1, layer-wide barriers). Run: `./m0-a100/run_lambda.sh ubuntu@<ip>`.
Workload: 16 layers of a Llama-1B-shaped MLP decode at batch 1 (1.61 GB of bf16 weights per forward). Peak HBM about 1555 GB/s.

**Proven on Ampere (no TMA, no setmaxnreg, no hardware CLC):**
- Persistent kernel with per-task dependency counters, fence-before-signal, and a dynamic claim queue is **correct**: bitwise identical to the per-op baseline and a CUDA graph, in static and dynamic modes, and CPU-reference PASS.
- Fences and counters cost essentially nothing (removing the fence changed timing by noise).
- With dependency waits skipped (timing probe only, wrong results) the same kernel reaches **1195 us = 87% of peak HBM**, faster than the CUDA graph (1400 us, 74%). So the task loop and loads are sound.

**Not yet a speedup:**

| variant | base+graph | mega-static | mega-dynamic |
|---|---|---|---|
| v1 (2 barriers/layer), 2 blocks/SM | 1401 us | 1574 us (0.89x) | 1541 us (0.91x) |

v2 makes the per-op baseline slower too (tiny tasks), so its ratio improves more than its absolute time; v1 remains the best absolute megakernel so far. v2 absolute times were not captured in the run output.
| v2 (split-K, chunked deps), 2 blocks/SM | 1651 us | 0.88x | 0.95x |

**Diagnosis.** The whole gap is the dependency barriers (~11 us each, about one task duration, a tail effect), not launch cost or fences. At batch 1 this MLP chain has inherent all-to-all barriers (rmsnorm needs all of x; down needs all of act), and a graph pays the same ones. Next levers, in order:
1. Interleave two independent streams (drafter+verifier, or two requests) so one fills the other's tail. This is the "share the GPU" idea and the strongest test of the megakernel thesis; compare against a 2-stream graph.
2. Cut per-task serial overhead in v2 (101 regs caps it at 2 blocks/SM).
3. Port the same VM to sm_90/sm_100 (TMA, `setmaxnreg`, CLC), where Hazy measured the gain.

Honest read: on A100 at batch 1 the megakernel is within about 5 to 10% of a CUDA graph and correct; it does not win yet. The 87% probe shows the headroom is real but sits in the barrier structure.

---

## 11. M1 demo: training + inference + a game in ONE kernel (A100)

Code: [`m1-demo/vm.cu`](m1-demo/vm.cu). Run: `./m1-demo/run_lambda.sh ubuntu@<ip>` (add `--live --all-only` over `ssh -t` to watch Pong in the terminal).

One persistent kernel, 216 blocks (2 per SM), three job streams with different contracts:
- **Game**: Pong physics plus a rasteriser, paced at 60 FPS (16.7 ms frame deadline), frames written straight into pinned host memory.
- **Inference**: the 8-layer decode chain from M0, paced at a token rate.
- **Training**: a 64x1024x1 tanh MLP trained with SGD (forward, backward, update all inside the kernel, fixed reduction orders), unpaced filler.

Scheduler: priority game > inference > training. A block claims a job's head task only if its dependencies are already satisfied, by CAS on a 32-way sharded ticket, so a block never blocks mid-task and preemption happens at task boundaries. Dependencies are monotonic global counters with a fence before every signal.

### Running the demo from your Mac

The kernel needs an NVIDIA GPU (built for A100, `sm_80`), so it builds and runs on the Lambda box; your Mac only drives it over SSH. Prerequisites: `ssh ubuntu@<ip>` works without a prompt (key in `~/.ssh/config`) and the box has `nvcc` and `rsync`.

```bash
cd flapping-airplanes/m1-demo

# full benchmark: solo vs concurrent, determinism checks, comparison against two CUDA-graph streams (about 1 minute)
./run_lambda.sh ubuntu@<ip> --secs 4

# watch it live: Pong in your terminal while inference and training run in the same kernel (needs a real terminal)
./run_lambda.sh ubuntu@<ip> --secs 20 --live --all-only
```

`run_lambda.sh` rsyncs the folder to `~/flapping-m1` on the box, compiles `vm.cu` with `nvcc -arch=sm_80`, and runs it over `ssh -t`. Useful flags: `--secs N` (seconds per run), `--tok-us N` (token period in microseconds; 3000 = 333 tokens/s), `--live`, `--all-only` (skip the solo runs and comparison), `--det-steps N`. Debug env vars: `MK_TPROF=1` (per-task training timeline), `MK_TBENCH=1` (per-phase kernel times).

Results on one A100-SXM4-40GB (4 s runs, current `vm.cu`):

| check | result |
|---|---|
| game deadline misses (60 FPS, 16.7 ms) with inference + training running | **0**; frame latency p99 25 us |
| tokens at 333/s with game + training running | token p99 0.93 ms (solo 1.03 ms); training keeps 78% of its solo throughput (15.1k of 19.4k steps/s) |
| inference output, solo vs concurrent | **bitwise identical** |
| training weights after exactly 400 steps: solo, concurrent, per-op CUDA graphs | **bitwise identical** (all three); also identical at every task size tried |
| training | loss 0.107 to about 0.0001 in 4 s (overfits its 4096-sample set) |

**Interleave gain, unpaced (1500 tokens + training, vs the same jobs as two CUDA-graph streams, inference on the high-priority stream):**

| | tokens/s | train steps/s | fraction of solo capacity, summed |
|---|---|---|---|
| 2-stream graphs, both | 1201 | 6528 | 0.98 + 0.19 = **1.17** |
| megakernel, both | 1120 | 9046 | 0.98 + 0.39 = **1.37** |
| solo, graphs | 1226 | 34417 | |
| solo, megakernel | 1165 | 23094 | |

**Honest read.** One persistent kernel serving tokens, training and a game gives training 1.39x the baseline's throughput while inference keeps 0.93x of the baseline's tokens/s (run-to-run noise is about 5%). Alone, inference is within 5% of a CUDA graph (1165 vs 1226 tok/s) and training reaches 0.67x of its graph (23.1k vs 34.4k steps/s; the graph runs the same task code, so every task-level speedup helps both). Per-op graphs still win on pure single-job speed; the megakernel's edge is running everything at once on one GPU without starving the background job.

**A100 optimisation log.** Each step was found by measuring, not guessing (device-side cycle counters, then per-task timestamps).
1. *Single global ticket per job*: claims serialise at about 1 per us, so a 16,384-task token took 14 ms. Fix: 128 sharded tickets, one home shard per block. Inference 0.62x to 0.67x.
2. *Scheduler probe cost*: a probe round took about 10k cycles (runtime integer divisions, 12 candidates per lane spilling to local memory). Fix: compile-time layer count, one candidate per lane (32-shard window around the home shard), global scan only when idle. 12k to 1.9k cycles; inference to 0.89x.
3. *Smem reuse*: keep the normalised x / activation chunk in shared memory across consecutive tasks of the same token, layer and chunk. Removes per-task rmsnorm barriers; inference to 0.93x.
4. *Training task code*: the kernel's own tasks were slow (P1 30 us, P3 28 us). Coalesced feature-major input loads, and a gradient loop that reuses each load across 4 units, with the exact summation order kept (hash unchanged). Training 7.9k to 11.4k steps/s.
   Later rounds vectorised the shared-memory reads (float4, same summation order) and issued all independent loads before the first barrier: step 52 us to 43 us, training alone to 23.1k steps/s.
5. *Fuse the output phase away*: P1 threads already hold the hidden activations, so they emit partial output sums and every P3 task recomputes the error from them. Step goes from 3 phases to 2 (new hash, still bitwise equal across all variants).
6. *CAS pile-up*: per-task timestamps showed barrier gaps were only 1 us, but a phase's tasks started over 24 us because all 216 blocks picked the lowest ready shard and failed down the list one round trip at a time. Starting each block's search at a different lane: training alone 11.9k to 19.4k steps/s (step 87 us to 52 us), and training under load 4.1k to 8.4k.

**Negative results worth keeping.**
- *Loader-warp design* (`vm3_loader.cu`): 8 compute warps plus 1 loader warp that `cp.async`-streams weights into two 64 KB smem pages, with deadlock-safe claim-ahead across dependency barriers. Correct and bitwise identical but slower (about 550 tok/s). With 1 block per SM and 2 pages the compute warps idle 84%. This is the design that should pay off with TMA, mbarriers and `setmaxnreg` on Hopper, where a loader is nearly free; untested (no H100/H200 access).
- *Smaller training tasks* (2 to 8 units per task) are slower, because every P3 task repeats the partial-sum prologue.

Files: `m1-demo/vm.cu` (current best), `vm2.cu` (earlier sharded-ticket version), `vm_3phase.cu` (before the phase fusion), `vm3_loader.cu` and `vm3_loader_prof.cu` (loader-warp experiment, the latter with cycle counters). Set `MK_TPROF=1` for the per-task training timeline and `MK_TBENCH=1` for per-phase kernel times.

The "game" is Pong with scripted paddles; training does not yet drive the game (no RL loop), and inference is a weight-streaming MLP chain, not a full transformer.

---

## 12. Live demo: one kernel, three programs, in your browser (`live-demo/`)

Modelled on the structure of the original Flapping Airplanes demo video (one dashboard with a training panel and its loss curve, an inference panel streaming text, a game panel, a real-time clock, then a comparison against separate kernels). Everything on screen is measured from the running kernel; nothing is scripted.

**What runs in the single kernel launch (A100, 216 resident blocks):**
- **Training**: a character-level language model (8-character context, 512 tanh units, 96-way softmax), forward, backward and SGD inside the kernel. About 14k steps per second at batch 256.
- **Inference**: text generated by the same live weights as they train, so you watch gibberish turn into sentences while the loss falls. Paced at 1000 tokens per second for readability.
- **Game**: **real Doom** (the doomgeneric port, shareware `doom1.wad`, E1M1) running its own game loop and its own software renderer inside the kernel, one tic per frame, on a single GPU thread (about 24 ms per frame standalone, about 31 fps of game time when sharing the GPU with the other two jobs). A scripted bot plays: god mode and all weapons via the real cheat codes, then walks, turns, strafes, shoots and opens doors. Its clock is virtual (one tic per frame), so the game is deterministic. See `doom-gpu/` for how the C engine is compiled for the GPU.
- **Scheduler**: the sharded-ticket claim queue from section 11, with deadlines and pacing per job.

**Run it from your Mac** (SSH to the GPU box must work without a prompt):
```bash
cd flapping-airplanes/live-demo
../doom-gpu/setup_box.sh ubuntu@<ip>   # ONCE: installs clang with sudo apt, clones doomgeneric, fetches the shareware WAD (md5-checked), builds the engine for the GPU
./run_demo.sh ubuntu@<ip>            # builds on the box, tunnels port 8080 to your Mac, opens http://localhost:8080
./run_demo.sh ubuntu@<ip> 8080 --secs 45 --cmp-secs 8     # longer live phase
```
The live phase is 30 s by default, then the comparison runs (megakernel with unpaced inference, then the same work as per-op kernels in CUDA graphs on two streams, each stream driven by its own host thread), then the results screen stays up. Ctrl-C in the terminal stops everything.

**Measured on this A100** (training + inference + game in the megakernel vs the same training + inference as per-op CUDA graphs on two streams, each driven by its own host thread; three consecutive runs agreed within 1%): megakernel 13.8k train steps/s, 53.3k tokens/s and 60 fps; separate kernels 14.5k steps/s and 55k tokens/s with no game. So the separate kernels are about 5% faster at training and 3% at inference, the megakernel launches 1 kernel against roughly 5 to 6 hundred thousand for the same work, and only the megakernel runs the game. Training alone: 14.6k steps/s in the megakernel vs 15.9k for the per-op graph (0.92x). The real demo's "34% slower" baseline is not reproduced here: the model is tiny (the GPU is mostly idle, which the panel labels report as SMs busy) and the hardware is an A100, not a B200.

**Optimisation log for this demo** (all measured; the step went from about 115 us to about 70 us):
1. *Slow task code.* P3 was 127 us (a serial 256-step load chain plus a serial bias-gradient sum), now 36 us; P2's softmax tail is spread over the 16 threads that own each sample, and P2 is register-blocked (4 samples x 6 logits per thread) so shared-memory wavefronts, not FMAs, stop being the limit; inference B's softmax and sampling run in one warp (18 to 15 us). Both the megakernel and the baseline run this same task code and both got faster.
2. *CAS claims serialise.* Per-task timestamps showed barrier gaps of only 1 to 2 us but tasks inside a phase finishing over 25 us. A compare-and-swap claim lets one block win per memory round trip; training now claims with one `fetch-add` per block on a per-(step, phase) counter (counter slots recycled by the step's last finisher; a step can only be cancelled at its start, by CAS to a dead value). Training alone 9.7k to 10.7k steps/s.
3. *Two blocks on one SM.* Phases have fewer than 108 tasks, but with 2 resident blocks per SM about 19 task pairs landed on the same SM and ran 40% slower, and the slowest sets the phase length. Only the first resident block of each SM claims training tasks. Training alone 10.7k to 14.6k steps/s.
4. *Things that did not work:* rotating each block's start offset to avoid L2 hot spots (no gain, the modulo made `dh` slower; reverted), and more loads in flight for the weight-gradient loop (no gain; its remaining cost is global-load throughput, which needs shared-memory staging).
Summation order changed in P2 (group-split reduction, fixed order) and its softmax, so results differ from the earlier build in the last bits; they are still deterministic for a given step count. The bitwise solo-vs-concurrent checks live in `m1-demo/vm.cu`, not in this demo, and were not re-run here. Debug switches: `DEMO_TBENCH=1` (per-phase kernel times; add `-DSECT` for in-task section cycle counts), `DEMO_SOLO=1` (training alone, megakernel vs graph, with a per-task timeline).

**What was found on the way (all fixed):** a baseline that throttled inference to the training rate, a corpus the model memorised in 2 seconds (flat loss curve), P3's 127 us task (a serial 256-step load chain; now 44 us), and a hung "kernel time" caused by one task doing a serial sum.

### Doom on the GPU (`doom-gpu/`)

The engine is plain C99 (about 60k lines, `ozkl/doomgeneric`, GPL). It is compiled with clang's NVPTX backend (not nvcc) and linked into the megakernel as a device object. What it took:
- **A tiny libc** (`tinylibc/`, `tinylibc.c`): printf family, string and ctype functions, a bump allocator, and an in-memory `fopen` that serves the WAD from device memory. Debug output goes to pinned host memory so it survives a crash.
- **Variadic functions at the IR level** (`fix_varargs.py`): clang 14's NVPTX backend cannot select `va_start`/`va_arg`, so every variadic function is rewritten to take an explicit packed-argument pointer and every call site packs its extra arguments into a stack buffer.
- **Source patches** (in `build_doom.sh`, applied to a working copy): PTX cannot statically initialise a pointer to an array element (`finecosine`, two input arrays) or self-referential tables (the sound table, two LLVM-generated lookup tables), so those are set at startup or avoided.
- **Linking**: PTX is assembled with `nvcc -dc -maxrregcount=112` and linked with the megakernel. The cap keeps the megakernel at 128 registers (two blocks per SM); without it a Doom function wanted 156.
- **Per-thread stack**: 24 KB (the limit is reserved for every resident thread: 221k threads x 24 KB is about 5 GB).
- **Time**: Doom spins in `TryRunTics` waiting for its 35 Hz clock, so `DG_SleepMs` advances a virtual clock instead of sleeping.

Standalone check (`doomtest.cu`): boot takes 0.9 s, 120 frames at 23.9 ms each, screenshot dumped from the GPU matches Doom's E1M1 start. Sound is disabled; the shareware WAD and the engine source are fetched by `setup_box.sh` and are not stored in this repository.

