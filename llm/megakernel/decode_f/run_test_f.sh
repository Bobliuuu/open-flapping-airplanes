#!/bin/bash
# on the box inside ~/llm/megakernel/decode_f : builds and checks decode for config F
set -e
D=${1:-/tmp/decode_data_F}
nvcc -arch=sm_80 -O3 -std=c++17 -lineinfo -o decode_test_f test_decode_f.cu
[ -f $D/weights.bin ] || python3 ref_decode_f.py gen $D 32
flock -w 7200 ~/.gpu.lock ./decode_test_f $D
python3 ref_decode_f.py check $D
