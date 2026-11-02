<div align="center">

<img src="site/assets/hero.svg" alt="OFA: one kernel, three programs" width="100%">

**🧠 trains &nbsp;·&nbsp; 💬 serves &nbsp;·&nbsp; 👹 plays Doom &nbsp;·&nbsp; all at once, in a single GPU kernel launch**

![GPU](https://img.shields.io/badge/GPU-NVIDIA%20A100-76b900?logo=nvidia&logoColor=white) ![launches](https://img.shields.io/badge/kernel%20launches-1-27c8e8) ![doom](https://img.shields.io/badge/Doom-35%20fps%20on%20the%20GPU-f2c14e) ![status](https://img.shields.io/badge/status-research%20demo-lightgrey)

[**📊 Stats site**](site/) &nbsp;·&nbsp; [**🏗️ Architecture**](#️-architecture) &nbsp;·&nbsp; [**🚀 Quickstart**](#-quickstart) &nbsp;·&nbsp; [**📈 Results**](#-results) &nbsp;·&nbsp; [**⚠️ Caveats**](#️-honest-caveats)

</div>

---

**open-flapping-airplanes** puts three unrelated programs inside **one persistent GPU kernel** on a single NVIDIA A100 and runs them at the same time:

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
    G[demo.cu megakernel] -->|nvcc -rdc=true| H((ofa demo binary))
    F --> H
    H -->|pinned memory + HTTP| I[browser: 3 panels]
```

## 🚀 Quickstart

You need an NVIDIA A100 box (sm_80) reachable over SSH. Run these on your Mac or Linux machine:

```bash
./doom-gpu/setup_box.sh ubuntu@<ip>          # once: clang, doomgeneric, shareware WAD (md5 checked)
./a100-demo/run_demo.sh ubuntu@<ip>          # build, run, tunnel, open http://localhost:8080
./benchmarks/run_benchmarks.sh ubuntu@<ip> 5 6   # 5 reps x 6 s, writes benchmarks/results/
python3 -m http.server 8000 --directory site     # the stats site, http://localhost:8000
```

No A100 box? The same demo and benchmark run on a [Modal](https://modal.com) B200 with no SSH (details in [`benchmarks/README.md`](benchmarks/README.md)):

```bash
uv tool install modal && modal setup                 # once
modal run benchmarks/run_modal.py --mode bench       # builds on CPU, then benchmarks on one B200 (about 7 GPU-minutes)
```

## 📈 Results

NVIDIA A100-SXM4-40GB (108 SMs), 5 repetitions x 6 s, 2026-11-01. Source: [`benchmarks/results/latest.json`](benchmarks/results/latest.json) and [`summary.md`](benchmarks/results/summary.md).

> 🎯 **open-flapping-airplanes, all three at once (mean, sd):** **13,507.6** training steps/s (sd 214.1) · 3,457,953 training tok/s · **53,211.6** inference tok/s (sd 751.2) · **35.0** Doom fps (sd 0.0) · **1** kernel launch

The baselines run the *same task code* as ordinary per-op kernels (4 launches per step, 3 per token), each job **alone** unless stated. Doom has no per-op baseline.

| Baseline | Train steps/s | Infer tok/s | Baseline slower than open-flapping-airplanes, train / infer | 4,000 steps take |
|---|---|---|---|---|
| Per-op kernels, eager launches | 15,282 | 52,070 | -13.1% / 2.1% | 0.262 s |
| Per-op kernels, CUDA graphs | 16,296 | 60,343 | -20.6% / -13.4% | 0.245 s |
| Both jobs, one process, two CUDA-graph streams | 14,960 | 53,377 | -10.8% / -0.3% | 0.267 s |
| 🥊 Two separate processes sharing the GPU | 7,465 | 27,278 | **44.7% / 48.7%** | 0.536 s |

Negative means the baseline is faster. open-flapping-airplanes takes 0.296 s for 4,000 steps.

**The same benchmark on an NVIDIA B200** (148 SMs, on Modal, 5 repetitions x 6 s, 2026-11-01). Source: [`b200_20261101.json`](benchmarks/results/b200_20261101.json) and [`summary_b200.md`](benchmarks/results/summary_b200.md).

> 🟩 **open-flapping-airplanes on a B200, all three at once (mean, sd):** **17,788.6** training steps/s (sd 332.7) · 4,553,879 training tok/s · **71,349.9** inference tok/s (sd 382.4) · **35.0** Doom fps (sd 0.0) · **1** kernel launch

| Baseline (B200) | Train steps/s | Infer tok/s | Baseline slower than open-flapping-airplanes, train / infer | 4,000 steps take |
|---|---|---|---|---|
| Per-op kernels, eager launches | 22,239 | 68,712 | -25.0% / 3.7% | 0.180 s |
| Per-op kernels, CUDA graphs | 22,111 | 66,873 | -24.3% / 6.3% | 0.181 s |
| Both jobs, one process, two CUDA-graph streams | 21,450 | 64,109 | -20.6% / 10.1% | 0.186 s |
| 🥊 Two separate processes sharing the GPU | 10,328 | 29,961 | **41.9% / 58.0%** | 0.387 s |

open-flapping-airplanes takes 0.225 s for 4,000 steps on the B200. Its rates are about 1.3x the A100's, roughly the ratio of the maximum SM clocks (1,965 vs 1,410 MHz); the extra SMs stay idle. The per-op training baseline gained more (1.46x), so its lead grew.

### 🆚 Versus Flapping Airplanes' published numbers

Their column is **read frame by frame from their demo video** ([`benchmarks/fa_published.json`](benchmarks/fa_published.json)); we did not measure it.

**Like-for-like, at LLM scale on one B200** ([`b200-run/`](b200-run/README.md)): the same GPU and 116 / 24 / 8 split, a 162M-parameter GPT with a 100k-token vocabulary, and Megatron + Transformer Engine and vLLM as the baselines, each alone on its SM share.

| | Flapping Airplanes (B200, from video) | open-flapping-airplanes (B200, measured) |
|---|---|---|
| Kernel launches for the same work | 1 vs 514,562 | 1 vs 1.2M to 8.4M |
| 4,000 training steps | 46.1 s vs Megatron 70.1 s (baseline **34% slower**) | 95.8 s vs Megatron 62.4 s (baseline **54% faster**) |
| Serving | 1,076 vs vLLM 710 tok/s (baseline **34% slower**) | 1,764 vs vLLM 684 tok/s (baseline **61% slower**) |
| Doom | plays throughout | 35 fps throughout |

open-flapping-airplanes reproduces their serving result but not their training result: its GEMMs reach 13-33% of cuBLAS on the B200, Megatron's run at cuBLAS speed. Every comparison, and what is still not like-for-like: [`docs/FA_COMPARISON.md`](docs/FA_COMPARISON.md).

The small demo above, against the same per-op kernels:

| | Flapping Airplanes (B200, from video) | open-flapping-airplanes demo (B200, measured) | open-flapping-airplanes demo (A100, measured) |
|---|---|---|---|
| Kernel launches | 1 vs 514,562 | 1 vs 4 per step / 3 per token | 1 vs 4 per step / 3 per token |
| 4,000 training steps | 46.1 s vs 70.1 s (baseline **34% slower**) | 0.225 s vs 0.180 s eager (baseline **25.0% faster**) | 0.296 s vs 0.262 s eager (baseline **13.1% faster**) |
| Inference tok/s | 1,076 vs 710 (baseline **34% slower**) | 71,350 vs 68,712 eager (baseline 3.7% slower) | 53,212 vs 52,070 eager (baseline 2.1% slower) |
| SM split | 116 train / 24 infer / 8 Doom of 148 | dynamic, mean busy SMs 17.1 / 2.9 / 0.14 of 148 | dynamic, mean busy SMs 18.5 / 2.3 / 0.19 of 108 |
| Model | LLM-scale (loss starts near 11.5) | tiny char-level MLP | tiny char-level MLP |

Their baselines run on their SM shares. With ours limited to the same shares by CUDA MPS (116 SMs for training, 24 for serving), the B200 training baseline is unchanged and the serving baseline gets 17% faster, so it beats open-flapping-airplanes at serving too: 22% faster at training, 17% faster at serving ([`b200_20261101_mps_baselines.json`](benchmarks/results/b200_20261101_mps_baselines.json)).

## ⚠️ Honest caveats

- 🖥️ **Same chip, model class and baselines as theirs, not their code.** [`b200-run/`](b200-run/README.md) matches their GPU, split and frameworks; their model size, data and settings are unpublished ([`docs/FA_COMPARISON.md`](docs/FA_COMPARISON.md)).
- 🔬 **The demo above is a tiny char-level model, not an LLM.** Its absolute throughput is not comparable with theirs; `b200-run/` is the LLM-scale version.
- 🎯 **Our per-op baseline is tight.** It reuses open-flapping-airplanes's own task code, so on this workload it is actually *faster per job* (13% to 21% on training on the A100, 24% to 25% on the B200). open-flapping-airplanes's advantage here is running all three at once: two processes that share a GPU are 45-49% slower on the A100 and 42-58% slower on the B200.
- ❌ **We do not reproduce their 34% at training.** On the same B200, Megatron + Transformer Engine is 54% faster than open-flapping-airplanes's megakernel, whose GEMMs are far from cuBLAS speed; at serving, open-flapping-airplanes beats vLLM by 61%.
- 💤 **The GPU is mostly idle for our workload** (about 21 of 108 SMs busy on average on the A100, about 20 of 148 on the B200): this demonstrates the mechanism, it is not a saturation benchmark.
- 🤖 **Doom is played by a script**, and 🔇 **sound is disabled**.

## Demo video

[Watch the explainer (86 s, `video/release/ofa_explainer_preview.mp4`)](video/release/ofa_explainer_preview.mp4). It plays when opened from the repo on GitHub. The B300 numbers in it are interim.

## 🗺️ Repo map

| Path | What |
|---|---|
| `a100-demo/` | A100 small-model Doom demo: `demo.cu` megakernel with embedded HTTP server, `index.html` three-panel dashboard, `run_demo.sh` |
| `doom-gpu/` | Compiling the C Doom engine for the GPU: tinylibc, `fix_varargs.py`, `gpu_par.c`, `build_doom.sh`, `setup_box.sh`, `doomtest.cu` |
| `benchmarks/` | `run_benchmarks.sh` (SSH box), `run_modal.py` (Modal B200, no SSH), `compare.py`, `fa_published.json`, `results/` |
| `site/` | Static stats website with animated diagrams (`index.html`, `data/`, `assets/`) |
| `llm/` | LLM + Doom open-flapping-airplanes build, baselines and the B200/B300 kit (to be regrouped into `a100-llm/` and `b300-run/`, see `docs/REORG_PLAN.md`) |
| `video/` | Sources of the explainer video |
| `tools/` | `gen_svgs.py` regenerates the animated SVGs |
| `docs/RESEARCH.md` | The original research log: feasibility analysis, build ladder, milestone notes |
| `b200-run/` | The LLM-scale megakernel against Megatron + Transformer Engine and vLLM on one B200 with Flapping Airplanes' protocol (`modal_app.py`, `results/`) |
| `docs/FA_COMPARISON.md` | Every open-flapping-airplanes run next to Flapping Airplanes' numbers: what is like-for-like (B200 vs B200) and what is not |

## 🔧 How Doom runs on a GPU

doomgeneric is compiled with **clang's NVPTX backend** against a small hand-written **tinylibc** (the engine needs only a few libc calls). clang 14 cannot select `va_start` for NVPTX, so `fix_varargs.py` **lowers varargs at the LLVM IR level**. PTX cannot emit some static initialisers (self-referential tables, pointers to array elements), so `build_doom.sh` patches those to be set at startup, and sound is disabled for the same reason. The object is linked into the CUDA kernel with relocatable device code, with a **register cap of 112** and a **24 KB per-thread stack**. Warp-0 divergence otherwise hangs a named barrier, so every participant meets at **one shared barrier instruction**.

## 📜 Credits and licensing

- **doomgeneric** by ozkl (GPL) is cloned by `setup_box.sh` and is not stored in this repo. **Doom** is by id Software.
- The **shareware `doom1.wad` is not included**: `setup_box.sh` downloads it and keeps it only if its md5 matches.
- Not affiliated with, endorsed by, or based on code from Flapping Airplanes.
- More background: [`docs/RESEARCH.md`](docs/RESEARCH.md), [`benchmarks/README.md`](benchmarks/README.md), [`site/README.md`](site/README.md).
