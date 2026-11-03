# llm/ofa_fast: training + inference + real Doom in ONE kernel, without the -rdc slowdown (A100)

Same program as `llm/ofa` (216 persistent blocks: 85 SMs train, 17 serve tokens, 6 run real Doom at 35 Hz, live dashboard), but built so that
the Doom build is as fast as the Doom-free `llm/megakernel` build.

```
MODEL=S ./run_ofa.sh ubuntu@<ip> 8080     # builds with build_merged.sh on the box, then runs under the GPU lock
MODEL=F ./run_ofa.sh ubuntu@<ip> 8080
# by hand, on the box (needs ~/doom-gpu/doom.ptx from doom-gpu/build_doom.sh):
./build_merged.sh ofa_fast_S -DMERGED_DOOM            # F: add -DMODEL_F
```

## Results (A100-SXM4-40GB, 85/17/6 split, 4,096 tokens/step)

| build | S, 4,000 steps | ms/step | notes |
|---|---|---|---|
| `llm/megakernel` fast (non-rdc, no Doom) | 189.3 s | 47.3-47.4 | reference |
| `llm/ofa` (rdc, Doom) | 224-226 s | 56.0-56.6 | the 18% regression |
| **`ofa_fast` merged (Doom ON)** | **190.7 s** | **47.67** | loss **2.1490**, decode **591 tok/s**, Doom **35.0 fps** (6,663 frames) |
| `ofa_fast` merged, `--no-doom`, 200 steps | | 47.30 | decode 599 tok/s |

* Loss steps 0..100 are byte-identical (`cmp` clean, `--loss-out`) between `ofa_fast`, `llm/ofa` and the megakernel fast build, with Doom on and with `--no-doom`.
* Model F, 250 steps, Doom ON: **102.5 ms/step** (39.96k tok/s), step-0 loss 11.8006, decode 324 tok/s, Doom 35.0 fps (`llm/ofa`: 114.9; perf fork without Doom: 104.65, so F is now faster than the Doom-free fork build).
* Against the Megatron-core + TE baseline restricted to 84 SMs (S: 233.2 s) the S run is **1.22x** faster (233.2/190.7); `llm/ofa` was 1.03x (226.2 s). Against the eager 226.2 s run it is **1.19x** (226.2/190.7).
* `dashboard_S_fast.png`: mid-run DevTools screenshot (step 416, loss 3.89, 588 tok/s decode, Doom 35 fps), all three panels live.

## Diagnosis

The 18% was `-rdc=true`, not Doom: the same `ofa.cu` built non-rdc with the Doom entry points stubbed runs 47.46 ms/step, and `llm/ofa` with the
`doom_loop` call stubbed out (still rdc) was as slow as with Doom.
Under `-rdc` nvcc passes `--device-c` to cicc and `--compile-only` to ptxas, so every non-inlined `__device__` function must obey the
relocatable ABI: `run_gemm` becomes an ABI function (240 B frame, 240 B spills; whole-program ptxas gives it 0 B / 12 B with a private
calling convention), `job_loop`/`run_train_task` are no longer cloned/inlined (1080 B and 456 B frames), and the kernel frame goes 552 B -> 1464 B.
Forcing everything inline in rdc is not enough (53.9-54.0 ms; `__forceinline__` on job_loop/run_train_task, `-DGEMM_INLINE`), and the same PTX
through ptxas without `--compile-only` is the same code, so the loss is in the ABI/inlining treatment, not in the PTX.
Doom needs the ABI only because it is a separate relocatable object.

## How the build works

`build_merged.sh` compiles `ofa.cu` **without** `-rdc`, but between cicc and ptxas it merges Doom's PTX into the kernel's PTX module:
1. `nvcc --dryrun` gives the exact steps for the normal non-rdc build (cudafe++, cicc, ptxas, fatbinary, host gcc, nvlink, link); the script replays them.
2. After the cicc step, `ptxmerge.py` takes the kernel PTX plus `~/doom-gpu/doom.ptx` (the clang-NVPTX output of `doom-gpu/build_doom.sh`) and writes one module.
   It drops the `.extern .func gpu_doom_*` declarations and the two data definitions (`I_VideoBuffer`, `colors`) that nvcc emits, and puts Doom's body in front.
3. ptxas compiles the merged module as one whole program (training code keeps the private calling convention; recursion inside Doom keeps its ABI).
4. `-DMERGED_DOOM` makes `doom_job.cuh` define host-side shadows of `I_VideoBuffer`/`colors` so the host registration links.
`doom_dev.o` and its `-maxrregcount=112` are no longer used: Doom's functions share the kernel's 128-register cap. Kernel (S): 128 regs, 2432 B frame
including the Doom call tree (F: 3088 B), 916/1256 B spills; the training hot path is unchanged in speed. The runtime stack limit is 24 KB with Doom and 4 KB
without (`--no-doom`), both above the frames; verified with `--no-doom` (47.30 ms/step, loss equal). If you add more stack to the kernel, keep 4096 above the
reported "cumulative stack size".

`ofa.cu` also gained `LR_ATTR/TASK_ATTR/JOB_ATTR` inline macros (defaults unchanged apart from `lr_at` being forceinline) and `doom_job.cuh` the `NO_DOOM_LINK`
stub section; they are only for the measurements above. `bld.sh` (ptxas -v of rdc variants), `tm.sh` (timing under the lock), `chk.sh`, `final.sh` are the measurement helpers.

## Fragility

* It depends on the `nvcc --dryrun` command layout (CUDA 12.8.93 here) and on the PTX text format: `ptxmerge.py` uses regexes for the extern declarations and the two globals.
  A different nvcc or a Doom rebuild that changes symbol names can break it. `build_merged.sh` prints the nvcc version, fails loudly if the cicc step is not found,
  the doom PTX is missing, any replayed step fails or no binary results, and `run_ofa.sh` repeats the nvcc version on failure.
* Doom's PTX is `.version 7.5`; merged into the 8.7 kernel PTX it assembles (it is the main module's version that wins).
* Doom is no longer a separately linked object: changing Doom means regenerating `doom.ptx` and rebuilding the whole kernel.
* The kernel frame includes Doom's recursion; its true stack need is bounded by the 24 KB limit set when Doom is on.
