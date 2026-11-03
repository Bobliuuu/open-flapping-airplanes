#!/bin/bash
# usage: ./bench.sh BIN [args]   -- runs a prebuilt binary on the box under the GPU lock (timing only, no persist test)
ssh ubuntu@<ip> "cd ~/llm/gemm2 && flock -w 7200 ~/.gpu.lock bash -c 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; ./$1 --nopersist ${@:2} | grep -v \"kernel L\"'"
