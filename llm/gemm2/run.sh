#!/bin/bash
# usage: ./run.sh CFG [test args]  -- syncs, builds gemm2_test_<CFG>, runs it (correctness; add --nobench to skip timing)
set -e
CFG=$1; shift
rsync -a --exclude 'gemm2_test*' ./ ubuntu@<ip>:~/llm/gemm2/
ssh ubuntu@<ip> "cd ~/llm/gemm2 && nvcc -arch=sm_80 -O3 -std=c++17 -DCFG=$CFG -Xptxas -v -o gemm2_test_$CFG test_gemm2.cu -lcublas 2>&1 | grep -E 'error|spill|Used' | sort | uniq -c | head -20; ./gemm2_test_$CFG $*"
