# b200/: Blackwell track of Triad (148-SM B200, sm_100a), prepared WITHOUT B200 hardware

> **Update 2026-11-02:** the megakernel (with Doom), the baseline scripts and the tcgen05 GEMM test have now run on an actual B200 (Modal
> `gpu="B200"`), unchanged; results and the pipeline are in [`../../b200-run/`](../../b200-run/README.md). The text below is the original plan.

**EVERYTHING HERE IS UNTESTED ON B200.** No B200 result exists. This kit was written and compile-checked on the A100 box (nvcc 12.8, which can target
`sm_100a`). Nothing below claims a measured Blackwell number. FA's published targets (from `benchmarks/fa_published.json`, read off their video, not measured by us):
Megatron 4,000 steps **70.1 s** (~230k tok/s) on 116 SMs, vLLM **710 tok/s** on 24 SMs, KVM 46.1 s / ~355k tok/s / 1,076 tok/s, 514,562 vs 1 launch, split 116/24/8 of 148.

| dir | content | state |
|---|---|---|
| `baseline/` | setup, MPS SM calibration, `run_all.sh` (Megatron+TE S/F, PyTorch, vLLM, launch counts), JSON in the A100 v2 schema, FA comparison | scripts syntax-checked only; python files parse; not run |
| `megakernel/` | SM-count-parameterised triad build (`--split auto`, 116/24/8 on 148), `build.sh sm_80|sm_100a` | sm_80 build + loss regression on A100 (see below); sm_100a compiles, never executed |
| `gemm_tcgen05/` | TMA + tcgen05 GEMM backend with the `gemm.cuh` interface | compiles for sm_100a (ptxas OK, SASS has UTCHMMA/UTMALDG); never executed; probably wrong somewhere (see its README, ranked risks) |

## What is tested (A100 box, nvcc 12.8.93, 2026-11-01)
| check | result |
|---|---|
| `megakernel/build.sh sm_80` (model S) | builds; `triad_kernel` 128 regs, 1472 B stack, 312 B spill stores / 364 B loads |
| 20-step loss, new sm_80 build vs a binary built from the unmodified `llm/triad/triad.cu` (training only, `check_loss.sh`) | **per-step loss files byte-identical** (step 0 = 11.710649 ... step 19 = 8.821497); run under `flock`. New build prints `108 SMs ... 216 persistent blocks`, smem/SM 167,936 B, opt-in 166,912 B, probe saw 108 SMs, max smid 107 |
| `build.sh sm_100a` S, mma backend | compiles + links (nvcc 12.8); `triad_kernel` 128 regs, 192 B stack, 376 B spill stores / 356 B loads, 21 nvcc warnings (pre-existing style) |
| `build.sh sm_100a` F (`MODEL=F`) | compiles + links; 128 regs, 1056 B stack, 1704 B spill stores / 1828 B loads |
| `build.sh sm_100a ... -DGEMM_STAGES=6` | compiles + links (96 KB/block) |
| `build.sh sm_100a ... -DTRIAD_GEMM_BACKEND=1` (S and F) | compiles + links; SASS contains UTCHMMA/UTMALDG (41 hits in S) |
| `build_doom_b200.sh sm_100a` | the existing `doom.ptx` assembles for sm_100a with nvcc 12.8 (4.3 MB object); never linked into a run on Blackwell |
| `gemm_tcgen05/test_gemm_sm100.cu` | compiles for sm_100a with backend 1 (all 8 kernel combos, 72-108 regs, 0 spills), backend 1 with `GEMM100_LAYOUT_MASK=2`, backend 0 on sm_100a, and backend 1 on sm_80 (falls back to mma.sync) |
| `baseline/` | `bash -n` / `ast.parse` on every script; `check_sms.cu` compiles for sm_100a. No Python was executed (no torch locally, no installs) |
| NOT run (the shared A100 lock was busy for >15 min) | `--split auto` on the A100 (should print 85/17/6 with Doom), and `test_gemm_sm100` backend-1-on-sm_80 fallback run |

## Runbook for the first B200 hours
Assumes one B200 (148 SMs), driver >= 570, CUDA toolkit >= 12.8 on PATH, this repo checked out (`llm/`, `benchmarks/`, `doom-gpu/` if Doom is wanted), user able to start MPS (`nvidia-cuda-mps-control`). Use `tmux`; keep one job on the GPU at a time (all scripts do).

### Hour 0: sanity (10 min)
```
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv      # expect B200, >=570, 10.0   (B300 = 10.3 -> use sm_103a everywhere below)
nvcc --version                                                            # >= 12.8
cd llm/b200/baseline && nvcc -arch=sm_100a -o /tmp/chk check_sms.cu && /tmp/chk   # expect device_reported_sms=148 distinct_smids_used=148
```
### Hour 1: baseline calibration (setup ~1 h of mostly waiting, then runs)
1. `cd llm/b200/baseline && ./setup_b200.sh` (venv-train: torch cu128, megatron-core, Transformer Engine built from source for sm_100 (30-60 min, `MAX_JOBS` limits RAM), flash-attn best effort;
   venv-serve: vLLM; TinyStories + cl100k tokenisation; MPS SM calibration scan). Order matters, see the header of `setup_b200.sh`. Re-run single steps with `STEPS="venv_train" ./setup_b200.sh`.
2. Read `~/b200_setup/sms_calibration.txt`: which `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` gives 116 / 24 SMs. Defaults are 78 and 16 (what FA's SM counts imply); override `TRAIN_PCT=.. INFER_PCT=..` if the scan says otherwise.
3. Smoke test (2 min): `STEPS=200 REPS=1 PHASES="count train_te" ./run_all.sh`. Check `results/runs_v2/environment.txt` (torch/TE/flash-attn versions) and the log for which attention backend TE chose (`NVTE_DEBUG=1`).
4. Full calibration: `./run_all.sh` (REPS=3: Megatron+TE S and F each full-GPU and 116-SM, PyTorch eager S/F, CUDA-graph variants, vLLM S/F full and 24-SM, launch counts). Each job runs alone (FA's method).
   Optional cross-check of the launch counts with Nsight Systems: `MODEL=S ./count_nsys.sh megatron`.
5. Outputs: `results/baseline_v2_b200_<date>.json` (same schema as `llm/baseline/results/baseline_v2_a100_*.json`) and `results/fa_compare_b200.md` (your numbers next to 70.1 s / 230k tok/s / 710 tok/s).

### Hour 2: the ported megakernel as-is (mma.sync backend, runs on Blackwell through the sm_100a compile)
```
rsync/copy data as for the A100 (llm/baseline/prep_data.py output, make_init_flat.py -> init_flat_S.f32 / _F, starts_4000.u32, make_vocab.py -> cl100k_vocab.bin, prompt_ids.txt)
cd llm/b200/megakernel
./build_doom_b200.sh sm_100a                    # only if doom-gpu/doom.ptx exists (else: NODOOM=1)
./build.sh sm_100a triad_mma                    # MODEL=F for F
# a) correctness first: training only, 20 steps; compare with the A100 golden losses (llm/baseline/golden/losses_0_20.json)
./triad_mma --steps 20 --data ~/data/tinystories_cl100k_train.u32 --init ~/data/init_flat_S.f32 --starts ~/data/starts_4000.u32 --loss-out /tmp/l20.txt
# b) all three jobs, FA's split (must print "SM partition (B200 preset 116/24/8 ...): infer 24, doom 8, train 116"):
./run_b200.sh ./triad_mma --steps 4000
```
Then try `./build.sh sm_100a triad_mma6 -DGEMM_STAGES=6` (uses the extra smem) and compare step time.
### Hour 3: tcgen05 backend (first the stand-alone GEMM test, only then the megakernel)
Follow the test plan in `gemm_tcgen05/README.md` (control run with backend 0, then `GEMM100_LAYOUT_MASK=2`, 3, 7; `timeout` every run; compare TFLOP/s with cuBLAS).
Only after all 15 S cases print PASS and bit-exact: `./build.sh sm_100a triad_tc -DTRIAD_GEMM_BACKEND=1`, rerun the 20-step loss check against `triad_mma`, then the 4000-step run.
### Hour 4: tune (in this order)
1. Phase table (`--prof-step`) of the mma and tcgen05 builds: where does a step spend time (GEMM vs attention vs elementwise vs scheduler gaps).
2. `GEMM_STAGES`, `GEMM100_STAGES`, `TRIAD_BLOCKS_PER_SM`, `--opt-mode`, `CLAIM_FETCHADD`.
3. tcgen05: smem-staged epilogue, epilogue/MMA overlap (double-buffered TMEM), 2-CTA (256x128), TMA multicast; attention backend on tcgen05 (not started; attention is mma.sync).
4. Split-K for the small TN GEMMs (36 tiles of dW_o underfill 148 SMs x 2 slots even more than 216).

## Expected failure points (in rough order of appearance)
1. Driver/toolkit: `no kernel image` or `PTX .target` errors = built for the wrong arch (use `sm_100a`, not `sm_100`, for tcgen05/TMEM code; plain `sm_100` or `sm_80` PTX JITs but has no tcgen05).
2. `torch.cuda.get_arch_list()` lacks sm_100 -> wrong torch wheel (needs cu128+). Transformer Engine source build fails on RAM (lower `MAX_JOBS`) or cuDNN headers (`CUDNN_PATH`).
3. flash-attn: no sm_100 kernels or no wheel; then TE must use cuDNN fused attention (`ATTN=fused ./run_all.sh`). FA says "flash-attn", so record which backend ran; throughput can differ a lot.
4. MPS: percentage rounding (calibration scan), or MPS refusing to start as non-root (`CUDA_MPS_PIPE_DIRECTORY` is user-level in the scripts); under MPS `max_compute_procs_seen` counts the daemon.
5. vLLM on Blackwell: wheel without sm_100 kernels, FlashInfer JIT compiling at first start (slow first run), CUDA-graph capture error; `--cg eager` (vLLM enforce-eager) is the escape hatch for a first number.
6. Megakernel: probe line says SMs != 148 (MPS leftovers? run on an idle GPU without MPS), launch rejected for smem/registers, Doom object link errors (rebuild with `build_doom_b200.sh`), Doom fps pacing off.
7. tcgen05: hang or garbage, see `gemm_tcgen05/README.md` R1-R4. Never run it without `timeout`.
8. Loss differs from the A100 golden in the 3rd-4th digit: expected for different GEMM accumulation orders; a NaN or a loss that does not fall is a bug.

## What to send back
* `nvidia-smi -q | head -30`, `nvcc --version`, `results/runs_v2/environment.txt`, `~/b200_setup/sms_calibration.txt`.
* `baseline/results/baseline_v2_b200_<date>.json` and `fa_compare_b200.md` (whole files).
* The megakernel: first 40 lines of stdout of each run (partition, smid probe, smem headroom), the final timing block (4000-step wall time, tok/s, infer tok/s, Doom fps), the `--prof-step` phase table, and the `ptxas` lines from `build.sh`.
* For any tcgen05 run: the test table (all rows), or the first failing row plus `compute-sanitizer` output; say which `GEMM100_LAYOUT_MASK` was used.
* Anything that hangs: which command, how long it ran, `nvidia-smi` utilisation.

## Cost guess (not a quote; check the provider's current price before launching)
One B200 on a specialised GPU cloud is roughly USD 4-8 per GPU-hour; hyperscaler 8-GPU B200 instances are priced per node (an order of magnitude more per hour, 8 GPUs) and you would use 1/8 of it. Rough GPU time: setup mostly CPU (1-1.5 h incl. TE compile),
full baseline `run_all.sh` REPS=3 about 1.5-2.5 h (24 training runs of ~1-2 min each plus 18 vLLM start-ups of 1-3 min each), megakernel hours 2-4 about 2-3 h. A first full session of 6-8 h is about USD 30-70 at those rates.
Use `REPS=1` and `PHASES=...` to cut it to ~2 h if the goal is just a first number.
