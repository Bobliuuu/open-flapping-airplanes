#!/usr/bin/env bash
# Usage (run on your Mac):  ./run_demo.sh ubuntu@<ip> [port] [extra demo args]
# Syncs the demo and the GPU Doom engine to the box, builds, starts, tunnels the dashboard to this Mac and opens it.
# First time only: ../doom-gpu/setup_box.sh ubuntu@<ip>   (installs clang, fetches the engine source and the shareware WAD)
set -euo pipefail
HOST="${1:?usage: run_demo.sh user@host [port] [demo args]}"; PORT="${2:-8080}"; shift || true; shift || true
DIR="$(cd "$(dirname "$0")" && pwd)"
rsync -az --exclude demo --exclude 'demo_*' "$DIR/" "$HOST:~/flapping-live/"
rsync -az --exclude doomgeneric --exclude obj --exclude src_gpu --exclude '*.o' --exclude '*.ptx' --exclude '*.bc' --exclude '*.ll' --exclude '*.cubin' "$DIR/../doom-gpu/" "$HOST:~/doom-gpu/"
( until curl -s "http://localhost:$PORT/state" >/dev/null 2>&1; do sleep 1; done; sleep 1; open "http://localhost:$PORT" ) &
ssh -t -L "$PORT:localhost:$PORT" "$HOST" "cd ~/doom-gpu && ./build_doom.sh >/dev/null && cd ~/flapping-live && nvcc -O3 -arch=sm_80 -std=c++17 -rdc=true demo.cu ~/doom-gpu/doom_dev.o -o demo -lpthread && ./demo --port $PORT --wad \$HOME/doom-gpu/doom1.wad $*"
