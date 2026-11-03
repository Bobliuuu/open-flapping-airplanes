#!/bin/bash
# runs on the box: ./bench_box.sh OUTFILE BIN [args]  -- waits for the GPU lock, runs one timing pass (<60 s) and writes OUTFILE
out=$1; bin=$2; shift 2
flock -w 7200 ~/.gpu.lock bash -c "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > $out; ./$bin --nopersist $* | grep -v 'kernel L' >> $out"
