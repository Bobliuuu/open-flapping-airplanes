#!/usr/bin/env bash
# Usage: ./run_lambda.sh ubuntu@<ip> [vm args]      (add -t style live demo: ./run_lambda.sh host --live --all-only)
set -euo pipefail
HOST="${1:?usage: run_lambda.sh user@host [vm args]}"; shift || true
DIR="$(cd "$(dirname "$0")" && pwd)"
rsync -az --exclude vm "$DIR/" "$HOST:~/flapping-m1/"
ssh -t "$HOST" "cd ~/flapping-m1 && nvcc -O3 -arch=sm_80 -std=c++17 vm.cu -o vm && ./vm $*"
