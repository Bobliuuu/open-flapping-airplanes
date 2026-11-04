# Research log: can a persistent megakernel own the GPU? (flapping-airplanes)

**Can we build what Flapping Airplanes says it built: a persistent "megakernel" runtime that owns the GPU's execution layer?**

Short answer: **yes for the serving half, probably for the rest.** This log is the feasibility case.

> Historical research log, written before the demo existed (sections 1-6 are the original feasibility analysis; section 7 records what was then built). See the [root README](../README.md) for the finished project. Original status note: research only.

---

## TL;DR

| | |
|---|---|
| **Who** | Flapping Airplanes: data-efficiency AI lab. CEO Ben Spector is first author of ThunderKittens and co-first author of Hazy Research's megakernels. $1B Series A at about $6B (Oct 21 2026). |
| **Claim** | "The world's most complete megakernel framework": training, inference and a video game in **one persistent kernel**, scheduled by a "Kittens VM (KVM)". |
| **Public evidence** | **None.** No code, paper or benchmark as of Oct 31. Only the X post and press. Their KVM name traces to a public ancestor: ThunderKittens branch `bvm-unity-pipeline` (May 2025). |
| **Feasibility** | Eight hardware requirements force the design. Public code already implements most of it (Hazy, Cohere, Cursor MoK, Lucebox, MPK). |
| **What stays theirs** | A deterministic training ISA, a multi-job preemptive scheduler, a model-to-instruction compiler. **None is needed to serve tokens.** |

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

---

## 3. Feasibility verdict, by component

| Component | Verdict | Why |
|---|---|---|
| Persistent kernel + per-SM instruction ring | **Feasible now** | Hazy / Cohere / MoK ship it |
| Dependency-counter dataflow (no grid barriers) | **Feasible now** | Same |
| Decode megakernel, B200/B300 | **Feasible now** | MegaKittens (`mk-v2-llama`), WIP but public |
| Decode megakernel, RTX Blackwell (sm_120) | **Feasible, port needed** | Lucebox (Apache-2.0, batch 1, DeltaNet + NVFP4 path). sm_120 has no TMEM; CLC work stealing exists |
| Model-to-instructions compiler | **Feasible, public starting points** | MegaKittens, MPK (OSDI 2026), Event Tensor |
| Bitwise-deterministic training instructions | **Feasible, hard** | MoK shows MoE training at 1.41x, bitwise. Backward + optimizer ISA is the real work |
| Multi-job preemptive scheduler (inference + training + game) | **Plausible, unproven** | No public reference. Preempt at task boundaries; first approximate with engine-level priority tiers |
| Their full unified system | **Unknown** | Nothing published. Treat as stretch goal, not baseline |

**Biggest risk is not the idea, it is synchronization bugs.** A kernel can be bitwise on every test-bed case and still diverge under real timing: data written through the generic proxy and released by flag to a TMA (async-proxy) consumer needs a `fence.proxy.async` first. A bitwise bed proves arithmetic, not synchronization. Checks that should exist before any timing counts:

- **Barrier accounting**: each opcode's arrive count equals its waiter's target.
- **CPU replay** of the descriptor stream (Hazy `python_vm.py`; AutoMegaKernel's static deadlock/race validator).
- **Fixed reduction order**, so outputs stay bitwise.
- **Determinism under real timing**: replay the same requests and compare bitwise against control runs.

---

## 4. Public code to start from

| Code | What | GPUs | Fit |
|---|---|---|---|
| [HazyResearch/Megakernels](https://github.com/HazyResearch/Megakernels), branch `mk-v2-llama` ("MegaKittens") | PyTorch fx graph to DAG optimizer to dispatcher to one persistent kernel; Llama-1B / 70B decode | B200 (sm100a), B300 (sm103a), CUDA 13+ | B300 decode. WIP |
| HazyResearch/Megakernels, `main` / `throughput` | Llama-1B batch-1 VM; Llama-70B TP throughput VM | H100 | Reference design |
| [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens), branch `bvm-unity-pipeline` | `prototype/vm`, `demos/kvm-runner` (May 2025) | H100 / B200 | Origin of the "KVM" name |
| [cohere-ai/cohere-megakernel](https://github.com/cohere-ai/cohere-megakernel) | Decode megakernel, 16 opcodes, counter barriers | H100 BF16 | RTX FP8/FP4 "soon"; no push since Sep 8 |
| [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub) `optimizations/megakernel` | Fused single-kernel decode, Qwen3.5-0.8B (DeltaNet + attention), NVFP4 path | RTX 3090; sm_120/121a | Closest public base for sm_120. Batch 1, grid-wide barriers |
| [mirage-project/mirage](https://github.com/mirage-project/mirage) branch `mpk` | HF model to megakernel compiler; 1.2 to 6.7x lower latency | Hopper; Blackwell demos | Compiler alternative. Active |
| RightNow-AI/AutoMegaKernel | Agent harness with statically checked schedule IR | sm_80/90/120 | Static-check idea; tiny models |
| ForgeMegakernel (arXiv, Sep 2026) | Coding agents + design milestones + mid-state test oracle | n/s | Agent method; 1.21x over SGLang |

Also: Cursor's Mixture-of-Kittens (MoE training megakernel, 1.41x, bitwise); CUTLASS SM120 example 87a (CLC tile scheduler on sm_120).

---

## 5. Open questions

1. **Flapping's code or numbers**: none as of Oct 31. Recheck before citing.
2. **MegaKittens beyond its Llama examples** (MoE decode ops): untested. RTX needs its own port (no TMEM on sm_120).
3. **Lucebox at batch > 1 and with MTP drafting**: unmeasured.

## 6. Their claims vs. what is public

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

## 7. Live demo: one kernel, three programs, in your browser (`a100-demo/`)

Modelled on the structure of the original Flapping Airplanes demo video (one dashboard with a training panel and its loss curve, an inference panel streaming text, a game panel, a real-time clock, then a comparison against separate kernels). Everything on screen is measured from the running kernel; nothing is scripted.

**What runs in the single kernel launch (A100, 216 resident blocks):**
- **Training**: a character-level language model (8-character context, 512 tanh units, 96-way softmax), forward, backward and SGD inside the kernel. About 14k steps per second at batch 256.
- **Inference**: text generated by the same live weights as they train, so you watch gibberish turn into sentences while the loss falls. Paced at 1000 tokens per second for readability.
- **Game**: **real Doom** (the doomgeneric port, shareware `doom1.wad`, E1M1) running its own game loop and its own software renderer inside the kernel, one tic per frame, on a single GPU thread (about 24 ms per frame standalone, about 31 fps of game time when sharing the GPU with the other two jobs). A scripted bot plays: god mode and all weapons via the real cheat codes, then walks, turns, strafes, shoots and opens doors. Its clock is virtual (one tic per frame), so the game is deterministic. See `doom-gpu/` for how the C engine is compiled for the GPU.
- **Scheduler**: the sharded-ticket claim queue (from the earlier prototypes, since removed), with deadlines and pacing per job.

**Run it from your Mac** (SSH to the GPU box must work without a prompt):
```bash
cd flapping-airplanes/a100-demo
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

