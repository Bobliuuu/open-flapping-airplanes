#!/usr/bin/env bash
# Run the full benchmark on a GPU box and pull the results back (run from your Mac):
#   ./run_benchmarks.sh ubuntu@<ip> [reps=5] [secs_per_measurement=6]
# Needs the one-time box setup first: ../doom-gpu/setup_box.sh ubuntu@<ip>
set -euo pipefail
HOST="${1:?usage: run_benchmarks.sh user@host [reps] [secs]}"; REPS="${2:-5}"; SECS="${3:-6}"
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$HERE/.."
rsync -az --exclude demo --exclude 'demo_*' "$ROOT/live-demo/" "$HOST:~/flapping-live/"
rsync -az --exclude doomgeneric --exclude obj --exclude src_gpu --exclude '*.o' --exclude '*.ptx' --exclude '*.bc' --exclude '*.ll' --exclude '*.cubin' "$ROOT/doom-gpu/" "$HOST:~/doom-gpu/"
STAMP="$(date +%Y%m%d)"
ssh "$HOST" "cd ~/doom-gpu && ./build_doom.sh >/dev/null && cd ~/flapping-live && nvcc -O3 -arch=sm_80 -std=c++17 -rdc=true demo.cu ~/doom-gpu/doom_dev.o -o demo -lpthread \
  && nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader \
  && ./demo --port 0 --wad \$HOME/doom-gpu/doom1.wad --bench /tmp/bench_$STAMP.json --reps $REPS --bench-secs $SECS"
mkdir -p "$HERE/results"
scp "$HOST:/tmp/bench_$STAMP.json" "$HERE/results/a100_$STAMP.json"
cp "$HERE/results/a100_$STAMP.json" "$HERE/results/latest.json"
python3 "$HERE/compare.py" "$HERE/results/a100_$STAMP.json"
echo "results: benchmarks/results/a100_$STAMP.json   site data: site/data/benchmarks.js (open site/index.html)"
