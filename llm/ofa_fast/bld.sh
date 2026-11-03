#!/usr/bin/env bash
# usage: bld.sh name "flags"  (on box, in ~/llm/ofa_fast)
N=$1; shift
nvcc -arch=sm_80 -O3 -std=c++17 --extended-lambda -rdc=true -maxrregcount=128 -Xptxas -v "$@" ofa.cu ~/doom-gpu/doom_dev.o -o ofa_fast_$N -lpthread 2>&1 | grep -A2 -E "Function properties for _Z12ofa_kernel|error" | grep -v nvlink
