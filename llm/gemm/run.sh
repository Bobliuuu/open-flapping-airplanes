#!/bin/bash
# usage: ./run.sh [test args]   (syncs, builds, runs on the A100 box)
set -e
rsync -a --exclude gemm_test --exclude '*.o' ./ ubuntu@<ip>:~/llm/gemm/
ssh ubuntu@<ip> "cd ~/llm/gemm && nvcc -arch=sm_80 -O3 -std=c++17 -Xptxas -v -o gemm_test test_gemm.cu -lcublas 2>&1 | grep -E 'error|spill' | sort | uniq -c | head; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; ./gemm_test $*"
