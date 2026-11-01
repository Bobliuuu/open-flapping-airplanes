#!/bin/bash
# usage: ./run_test.sh   (run on the box inside ~/llm/decode)
set -e
D=${1:-/tmp/decode_data}
nvcc -arch=sm_80 -O3 -std=c++17 -lineinfo -o decode_test test_decode.cu
[ -f $D/weights.bin ] || python3 ref_decode.py gen $D 32
./decode_test $D
python3 ref_decode.py check $D
