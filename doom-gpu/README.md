# doom-gpu

Builds the C Doom engine (doomgeneric) as a relocatable GPU object with clang's NVPTX backend. The engine source and the WAD are not stored here; `setup_box.sh` fetches them.

| File | Role |
|---|---|
| `setup_box.sh` | One-time: installs clang, clones doomgeneric, fetches the shareware WAD (md5 checked), builds |
| `build_doom.sh` | Compiles engine + tinylibc + glue to `doom.o`, applying the PTX-compatibility patches |
| `tinylibc/`, `tinylibc.c` | Minimal libc headers/implementation for device code |
| `fix_varargs.py` | LLVM IR pass lowering varargs (clang 14 cannot select `va_start` on NVPTX) |
| `gpu_glue.c`, `gpu_par.c` | Engine entry points; queued parallel column/span drawing for 255 workers |
| `doomtest.cu` | Standalone test: run Doom in one GPU thread for N frames and dump a frame |
