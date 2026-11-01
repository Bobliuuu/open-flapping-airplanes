#!/usr/bin/env bash
# Usage: ./run_lambda.sh ubuntu@<lambda-ip> [extra mk args]
# Syncs this dir to the Lambda box, builds for sm_80, runs correctness then timing.
set -euo pipefail
HOST="${1:?usage: run_lambda.sh user@host [mk args]}"; shift || true
DIR="$(cd "$(dirname "$0")" && pwd)"
rsync -az --exclude mk "$DIR/" "$HOST:~/flapping-m0/"
ssh "$HOST" "cd ~/flapping-m0 && nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv,noheader \
  && nvcc -O3 -arch=sm_80 -std=c++17 mk.cu -o mk \
  && echo '--- correctness ---' && ./mk --check \
  && echo '--- timing ---' && ./mk $*"
