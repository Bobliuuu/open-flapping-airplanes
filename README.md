<div align="center">

<img src="site/assets/hero.svg" alt="TRIAD: one kernel, three programs" width="100%">

**🧠 trains &nbsp;·&nbsp; 💬 serves &nbsp;·&nbsp; 👹 plays Doom &nbsp;·&nbsp; all at once, in a single GPU kernel launch**

![GPU](https://img.shields.io/badge/GPU-NVIDIA%20A100-76b900?logo=nvidia&logoColor=white) ![launches](https://img.shields.io/badge/kernel%20launches-1-27c8e8) ![doom](https://img.shields.io/badge/Doom-35%20fps%20on%20the%20GPU-f2c14e) ![status](https://img.shields.io/badge/status-research%20demo-lightgrey)

[**📊 Stats site**](site/) &nbsp;·&nbsp; [**🏗️ Architecture**](#️-architecture) &nbsp;·&nbsp; [**🚀 Quickstart**](#-quickstart) &nbsp;·&nbsp; [**📈 Results**](#-results) &nbsp;·&nbsp; [**⚠️ Caveats**](#️-honest-caveats)

</div>

---

**Triad** puts three unrelated programs inside **one persistent GPU kernel** on a single NVIDIA A100 and runs them at the same time:

| | Program | What happens |
|---|---|---|
| 🧠 | **Training** | A character-level language model learns online: forward, backward and SGD all run as tasks inside the kernel. |
| 💬 | **Inference** | Text streams out of the *live, still-training* weights, so the output visibly improves while the loss falls. |
| 👹 | **Doom** | The real engine (doomgeneric, shareware E1M1) runs its own game loop on the GPU, played by a scripted bot. |

> 🛰️ **Not affiliated with Flapping Airplanes.** Independent reproduction of an idea from their public demo video. We have not seen their code.

<div align="center">

![Live dashboard: training loss, streaming text, Doom](site/assets/dashboard.png)

*The demo is just these three panels. The numbers live on the [stats site](site/).*

</div>

## 🏗️ Architecture

<div align="center">

![Animated architecture](site/assets/architecture.svg)

</div>

**How one kernel runs three programs.** Every block is a long-lived worker. Warp 0 of each block polls sharded ticket queues, claims the next *ready* task (priority Doom > inference > training), and the other warps run it. Tasks signal completion through dependency counters guarded by memory fences, so a forward pass releases its backward pass and a token releases the next token, with no kernel launch in between.

```mermaid
sequenceDiagram
    autonumber
    participant S as warp 0 (scheduler)
    participant Q as ticket queues
    participant C as dependency counters
    participant W as compute warps
    loop persistent loop, inside ONE launch
        S->>Q: probe 32 shards (one memory round trip)
        Q-->>S: head task of each job
        S->>C: is its dependency met?
        C-->>S: counter >= target
        S->>Q: claim with CAS (Doom > infer > train)
        S->>W: run the task (registers + L2)
        W->>C: fence, then atomicAdd (signals dependents)
    end
```

### 👹 One Doom frame

<div align="center">

![Doom frame pipeline](site/assets/doom-pipeline.svg)

</div>

Thread 0 runs the game logic and *queues* column and span draws; at two flush points the other 255 threads of the block execute the queue behind a single named barrier. The picture is pixel-identical to serial drawing, but a frame drops from **24 ms to 6.7 ms**.

```mermaid
flowchart LR
    A[doomgeneric C source] -->|clang -target nvptx64| B[LLVM bitcode]
    T[tinylibc.c<br/>printf, string, WAD-in-memory] --> B
    B -->|llvm-link| C[one module]
    C -->|fix_varargs.py<br/>lowers va_start / va_arg| D[patched IR]
    D -->|llc| E[PTX]
    E -->|nvcc -dc, reg cap 112| F[device object]
    G[demo.cu megakernel] -->|nvcc -rdc=true| H((triad demo binary))
    F --> H
    H -->|pinned memory + HTTP| I[browser: 3 panels]
```

## 🚀 Quickstart

You need an NVIDIA A100 box (sm_80) reachable over SSH. Run these on your Mac or Linux machine:

```bash
./doom-gpu/setup_box.sh ubuntu@<ip>          # once: clang, doomgeneric, shareware WAD (md5 checked)
./live-demo/run_demo.sh ubuntu@<ip>          # build, run, tunnel, open http://localhost:8080
./benchmarks/run_benchmarks.sh ubuntu@<ip> 5 6   # 5 reps x 6 s, writes benchmarks/results/
python3 -m http.server 8000 --directory site     # the stats site, http://localhost:8000
```

## 📈 Results

NVIDIA A100-SXM4-40GB (108 SMs), 5 repetitions x 6 s, 2026-11-01. Source: [`benchmarks/results/latest.json`](benchmarks/results/latest.json) and [`summary.md`](benchmarks/results/summary.md).

> 🎯 **Triad, all three at once (mean, sd):** **13,507.6** training steps/s (sd 214.1) · 3,457,953 training tok/s · **53,211.6** inference tok/s (sd 751.2) · **35.0** Doom fps (sd 0.0) · **1** kernel launch

The baselines run the *same task code* as ordinary per-op kernels (4 launches per step, 3 per token), each job **alone** unless stated. Doom has no per-op baseline.

| Baseline | Train steps/s | Infer tok/s | Baseline slower than Triad, train / infer | 4,000 steps take |
|---|---|---|---|---|
| Per-op kernels, eager launches | 15,282 | 52,070 | -13.1% / 2.1% | 0.262 s |
| Per-op kernels, CUDA graphs | 16,296 | 60,343 | -20.6% / -13.4% | 0.245 s |
| Both jobs, one process, two CUDA-graph streams | 14,960 | 53,377 | -10.8% / -0.3% | 0.267 s |
| 🥊 Two separate processes sharing the GPU | 7,465 | 27,278 | **44.7% / 48.7%** | 0.536 s |

Negative means the baseline is faster. Triad takes 0.296 s for 4,000 steps.

### 🆚 Versus Flapping Airplanes' published numbers

Their column is **read frame by frame from their demo video** ([`benchmarks/fa_published.json`](benchmarks/fa_published.json)); we did not measure it.

| | Flapping Airplanes (B200, from video) | Triad (A100, measured) |
|---|---|---|
| Kernel launches | 1 vs 514,562 | 1 vs 4 per step / 3 per token |
| 4,000 training steps | 46.1 s vs 70.1 s (baseline **34% slower**) | 0.296 s vs 0.262 s eager (baseline **13.1% faster**) |
| Inference tok/s | 1,076 vs 710 (baseline **34% slower**) | 53,212 vs 52,070 eager (baseline 2.1% slower) |
| SM split | 116 train / 24 infer / 8 Doom of 148 | dynamic, mean busy SMs 18.5 / 2.3 / 0.19 of 108 |
| Model | LLM-scale (loss starts near 11.5) | tiny char-level MLP |

## ⚠️ Honest caveats

- 🖥️ **A100, not B200.** Different chip, different scale.
- 🔬 **A tiny char-level model, not an LLM.** Absolute throughput is not comparable with theirs.
- 🎯 **Our per-op baseline is tight.** It reuses Triad's own task code, so on this workload it is actually *faster per job* (13% to 21% on training). Triad's advantage here is running all three at once and beating two processes that share a GPU by roughly 45-49%.
- ❌ **We do not reproduce their 34%.** Their baseline is Megatron plus vLLM; ours is not.
- 💤 **The GPU is mostly idle for our workload** (about 21 of 108 SMs busy on average): this demonstrates the mechanism, it is not a saturation benchmark.
- 🤖 **Doom is played by a script**, and 🔇 **sound is disabled**.

## 🗺️ Repo map

| Path | What |
|---|---|
| `live-demo/` | The demo: `demo.cu` megakernel with embedded HTTP server, `index.html` three-panel dashboard, `run_demo.sh` |
| `doom-gpu/` | Compiling the C Doom engine for the GPU: tinylibc, `fix_varargs.py`, `gpu_par.c`, `build_doom.sh`, `setup_box.sh`, `doomtest.cu` |
| `benchmarks/` | `run_benchmarks.sh`, `compare.py`, `fa_published.json`, `results/` |
| `site/` | Static stats website with animated diagrams (`index.html`, `data/`, `assets/`) |
| `tools/` | `gen_svgs.py` regenerates the animated SVGs |
| `docs/RESEARCH.md` | The original research log: feasibility analysis, build ladder, milestone notes |
| `m0-a100/`, `m1-demo/` | Earlier persistent-kernel VM experiments on the A100 (history) |

## 🔧 How Doom runs on a GPU

doomgeneric is compiled with **clang's NVPTX backend** against a small hand-written **tinylibc** (the engine needs only a few libc calls). clang 14 cannot select `va_start` for NVPTX, so `fix_varargs.py` **lowers varargs at the LLVM IR level**. PTX cannot emit some static initialisers (self-referential tables, pointers to array elements), so `build_doom.sh` patches those to be set at startup, and sound is disabled for the same reason. The object is linked into the CUDA kernel with relocatable device code, with a **register cap of 112** and a **24 KB per-thread stack**. Warp-0 divergence otherwise hangs a named barrier, so every participant meets at **one shared barrier instruction**.

## 📜 Credits and licensing

- **doomgeneric** by ozkl (GPL) is cloned by `setup_box.sh` and is not stored in this repo. **Doom** is by id Software.
- The **shareware `doom1.wad` is not included**: `setup_box.sh` downloads it and keeps it only if its md5 matches.
- Not affiliated with, endorsed by, or based on code from Flapping Airplanes.
- More background: [`docs/RESEARCH.md`](docs/RESEARCH.md), [`benchmarks/README.md`](benchmarks/README.md), [`site/README.md`](site/README.md).
