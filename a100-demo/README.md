# a100-demo

`demo.cu` is one persistent CUDA kernel that trains a char-level LM, streams text from the live weights, and runs real Doom, while an embedded HTTP server feeds `index.html` (three panels: training, text, Doom).

```bash
../doom-gpu/setup_box.sh ubuntu@<ip>   # once
./run_demo.sh ubuntu@<ip> [port] [demo flags]   # syncs, builds, starts, tunnels, opens http://localhost:<port>
```

Flags (pass after the port): `--port N` (default 8080), `--secs S` (live run length), `--wad PATH` (doom1.wad), `--pace-us N` (inference pacing), `--start-delay S` (serve the dashboard S seconds before launching the kernel, for screen recordings), `--bench FILE` (run the benchmark suite and write JSON; used by `../benchmarks/run_benchmarks.sh`).
Build by hand: `nvcc -O3 -arch=sm_80 -std=c++17 -rdc=true demo.cu doom_dev.o -o demo -lpthread`.
The same source runs unchanged on a B200: build Doom with `ARCH=sm_100 ../doom-gpu/build_doom.sh` and the demo with `-arch=sm_100`, or let `../benchmarks/run_modal.py` do both on Modal.
