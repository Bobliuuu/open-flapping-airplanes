# b200/megakernel: the open-flapping-airplanes megakernel (train + decode + Doom, one launch) parameterised by SM count

Copy of `llm/ofa/` (ofa.cu, doom_job.cuh, dashboard.cuh, index.html, decode_wide.cuh, helper scripts) with these changes to `ofa.cu` only
(diff against `ofa_orig_reference.cu.txt`, the unmodified source):
* the SM partition comes from `cudaDeviceProp.multiProcessorCount` (nothing hard-coded; the old code only had 85/17/6 in `run_ofa.sh`),
* blocks per SM is a compile knob (`-DOFA_BLOCKS_PER_SM`, default 2) used for `__launch_bounds__`, the persistent block count and the utilisation report,
* shared-memory limits are read from the device (opt-in/block, per-SM, reserved) and printed; the launch refuses if the kernel does not fit the opt-in limit,
* a `%smid` probe builds a dense SM-rank table so the partition stays correct if `%smid` is not 0..SMs-1 (disabled SMs),
* the GEMM goes through `tg::` (`../gemm_tcgen05/gemm_backend.cuh`): mma.sync (default, identical code) or the tcgen05 backend (TMEM allocated once per block, 2 blocks per SM; the faster build on a B200).

## Partition
| flag / env | meaning |
|---|---|
| (none) | train only on all SMs (exactly the old default) |
| `--split auto` (or `OFA_SPLIT=auto`) | 108 SMs -> infer 17 / Doom 6 / train 85 (A100, proportional); 148 SMs -> 24 / **8** / 116 (B200, FA's split); other counts -> FA's proportions (24/148, 8/148, Doom >= 1) |
| `--split a100` / `--split b200` | force a preset (e.g. to emulate the B200 proportions on an MPS-limited GPU) |
| `--infer-sms N --doom-sms N [--train-sms N]`, env `OFA_INFER_SMS/DOOM_SMS/TRAIN_SMS` | explicit; train defaults to the remainder, a smaller value leaves SMs idle |
The chosen partition is printed at start (`SM partition (...)`) and goes to the dashboard. Doom keeps 8 SMs in the B200 split (16 blocks run the engine through its own ticket).
Roles are assigned per SM (dense rank of `%smid`): ranks [0,infer) decode, [infer,infer+doom) Doom, then train, then idle.

## Build
```
./build.sh sm_80                      # A100 (needs ~/doom-gpu/doom_dev.o as before)
./build_doom_b200.sh sm_100a          # re-assembles ~/doom-gpu/doom.ptx for Blackwell (clang NVPTX PTX is .target sm_80; nvcc 12.8 accepted it for sm_100a)
./build.sh sm_100a                    # B200, mma.sync GEMM        (MODEL=F ./build.sh sm_100a for config F)
./build.sh sm_100a ofa_tc2 -DOFA_GEMM_BACKEND=1 -DGEMM100_STAGES=2   # B200, tcgen05 GEMMs (3 stages: faster training, slower serving)
NODOOM=1 ./build.sh sm_100a           # without the Doom object (link stubs); then do not pass --wad
```
`build.sh` prints the ptxas line for `ofa_kernel`. Needs `-rdc=true` (Doom is a separate device object) and CUDA >= 12.8 for sm_100a (13.x also fine; for B300 use sm_103a).
Run: `./run_b200.sh ./ofa_b200_sm_100a --steps 4000` (data paths as in `llm/baseline`, vocab files from `make_vocab.py`; `GPU_LOCK=~/.gpu.lock` to serialise).

## Tuning knobs (all compile-time `-D` unless noted)
| knob | default | effect on B200 |
|---|---|---|
| `GEMM_STAGES` | 4 (64 KB) | mma.sync pipeline depth. B200 has 228 KB/SM: 2 blocks x 64 KB use 56%. `-DGEMM_STAGES=6` -> 96 KB/block, 2 blocks fit (192 KB + reserved). Build verified for sm_100a; speed unknown |
| `OFA_BLOCKS_PER_SM` | 2 | 1 frees 128 registers/thread budget (needs `-maxrregcount` change in build.sh) but halves latency hiding |
| `OFA_GEMM_BACKEND` | 0 | 1 = tcgen05 (see ../gemm_tcgen05): about 2x faster training on a B200 |
| `GEMM100_STAGES`, `GEMM100_LAYOUT_MASK` | 3, 7 | tcgen05 pipeline depth (2: 66,432 B of shared memory per block, the same L1 as mma.sync; 3: 99,200 B, faster training, less L1 for the serving SMs); layouts routed to tcgen05 (bit0 NN, bit1 NT, bit2 TN) |
| `OFA_CARVEOUT` (env) | unset | shared-memory carveout hint in percent (`cudaFuncAttributePreferredSharedMemoryCarveout`); serving is L1-sensitive |
| `CLAIM_FETCHADD` | 1 | scheduler ticket claim; 0 = old CAS loop (slower; 148 SMs make it worse) |
| `--opt-mode 0/1/2` (runtime) | 2 | AdamW overlap, as in MERGE.md |
| `OFA_PROBE_SMID=0` (env) | 1 | skip the %smid probe launch |
| `-DMODEL_F` / `MODEL=F` | S | config F (L4 d2048 H32); `make_init_flat.py` writes `init_flat_F.f32` |
Remember on a B200 the step is much shorter than on the A100, so scheduler overhead (tickets, dependency polling, `__nanosleep`) is a larger fraction: look at the phase table printed with `--prof-step`.

## What was verified (A100 box, nvcc 12.8; B200 NOT available)
See the table in `../README.md` ("What is tested"). In short: sm_80 build runs and its 20-step loss file is compared with a binary built from the unmodified source; sm_100a builds
(S, F, mma and tcgen05 backends) compile and link, with the ptxas stats listed there; nothing sm_100a has been executed.

## Expected B200 surprises (read before the first run)
1. `%smid` numbering or SM count differing from 148 (probe prints it; a mismatch warning means the partition is by rank, not by id).
2. `cudaLimitStackSize` 24 KB (Doom) x resident threads: 148 SMs x 2048 threads x 24 KB = 7.3 GB, fine on 180 GB but allocation takes a moment.
3. Doom's 35 Hz pacing uses `%globaltimer`; its resolution on Blackwell may be coarser (frame pacing jitter, not correctness).
4. If the first launch fails with "too many resources requested" or `invalid argument`: dynamic smem or register budget; run `build.sh` with `-DGEMM_STAGES=4` and `-DOFA_BLOCKS_PER_SM=1`.
5. `-DOFA_GEMM_BACKEND=1` hang (GPU stuck at 100%, no output): a block allocating TMEM per tile without relinquishing the permit serializes co-resident blocks; keep `tg::block_init`/`block_exit` around every block's tiles (see `../gemm_tcgen05/README.md`).
