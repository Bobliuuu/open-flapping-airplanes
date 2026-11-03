#!/bin/bash
# usage: ./sweep.sh "name:cfg:defines name2:cfg2:defines2 ..." [only]  -- build each variant in parallel on the box, then time --steady
# (or --only <substr>) under the GPU lock, all variants in ONE lock hold (keep <= 3 variants so the hold stays < 60 s)
rsync -a --exclude 'gemm2_*' --exclude 'r_*' ./ ubuntu@<ip>:~/llm/gemm2/
ssh ubuntu@<ip> "cd ~/llm/gemm2; for v in $1; do IFS=: read n c d <<< \"\$v\"; (nvcc -arch=sm_80 -O3 -std=c++17 -DCFG=\$c \$d -o gemm2_s_\$n test_gemm2.cu -lcublas 2>&1 | grep -E 'error' ) & done; wait; flock -w 7200 ~/.gpu.lock bash -c 'for v in $1; do IFS=: read n c d <<< \"\$v\"; echo == \$n; timeout 50 ./gemm2_s_\$n --steady --nopersist $2 | grep -v \"kernel L\" | cut -c1-24,90-; done'"
