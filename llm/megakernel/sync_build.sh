#!/usr/bin/env bash
# usage: ./sync_build.sh <binary-name> [extra nvcc flags...]   (builds ~/llm/megakernel/ofa_llm.cu on the box)
set -e
H=ubuntu@<ip>; BIN=${1:?}; shift || true
D="$(cd "$(dirname "$0")/.." && pwd)"
rsync -az --exclude 'golden' --exclude '*.f32' --exclude '*.u32' --exclude '*.txt' "$D/megakernel/" "$H:~/llm/megakernel/"
for a in gemm attention elementwise decode; do rsync -az "$D/$a/" "$H:~/llm/$a/"; done
ssh $H "cd ~/llm/megakernel && nvcc -arch=sm_80 -O3 -std=c++17 --extended-lambda -Xptxas -v $* -o $BIN ofa_llm.cu 2>&1 | grep -E 'error|ofa_kernel|spill' | head -8; ls -la $BIN | awk '{print \$5, \$9}'"
