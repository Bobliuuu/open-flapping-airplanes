# Planned reorganisation (not yet executed)

Deferred while these paths are still being edited. Use `git mv` so history follows.
Already done: `live-demo/` -> `a100-demo/`; `m0-a100/` and `m1-demo/` (Pong-era prototypes, no Doom) removed (tag `pre-trim` keeps them).
`doom-gpu/` stays as the shared Doom-engine build used by every demo (scripts also use the box path `~/doom-gpu`).

| Old path | New path | Note |
|---|---|---|
| `llm/ofa/` | `a100-llm/ofa/` | Doom + LLM open-flapping-airplanes on A100 (the real Doom build) |
| `llm/megakernel/` | `a100-llm/megakernel/` | no-Doom perf copy of the LLM kernel (Doom partition "not wired"); `MERGE.md` ports it into ofa. Candidate for deletion once merged: user's call |
| `llm/gemm/`, `llm/elementwise/`, `llm/decode/` | `a100-llm/kernels/{gemm,elementwise,decode}/` | building blocks |
| `llm/baseline/` | `a100-llm/baseline/` | A100 PyTorch/Megatron/vLLM baselines |
| `llm/gemm2/` | `a100-llm/kernels/gemm2/` | untracked, in progress |
| `llm/b200/` | `b300-run/` | kit written for B200, results were measured on a B300 (sm_103a); name keeps that honest |
| `llm/SPEC.md`, other `llm/*.md` | `a100-llm/` or `docs/` | shared spec |

## References to update when moving

- `video/tools/make_data.py`, `video/tools/refresh_numbers.py`, `video/data.json` ("source" strings): `llm/baseline/...` -> `a100-llm/baseline/...`, `llm/ofa/README.md` -> `a100-llm/ofa/README.md`, `b300-run/...` -> `b300-run/results/...`. `video/README.md` mentions `live-demo` (now `a100-demo`).
- `llm/b200/README.md`, `llm/b200/megakernel/{README.md,build.sh,build_doom_b200.sh,run_b200.sh}`: mention `llm/`, `benchmarks/`, `doom-gpu/` (box path `~/doom-gpu` unchanged).
- `llm/b200/baseline/count_nsys.sh` mentions "llm/baseline README".
- `llm/ofa/run_ofa.sh` and `llm/ofa/sync_build.sh`, `llm/megakernel/*.sh`: rsync/remote dirs `~/llm/<area>`; adjust if remote layout changes. `llm/ofa/README.md` and `llm/ofa/doom_job.cuh` comments mention `llm/baseline`, `doom-gpu/`.
- `llm/SPEC.md` mentions `live-demo/` and `llm/<area>` rules.
- Top-level `README.md` repo map: add `a100-llm/`, `b300-run/`.
