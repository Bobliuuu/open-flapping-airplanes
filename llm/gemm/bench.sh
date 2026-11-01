#!/bin/bash
# waits (up to ~2 min) for an idle GPU, then runs the timing build. usage: ./bench.sh [test args]
ssh ubuntu@<ip> "cd ~/llm/gemm && for i in \$(seq 1 40); do u=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits); [ \$u -lt 5 ] && break; sleep 3; done; echo util_before=\$u; ./gemm_test --nopersist $* | grep -v 'kernel L'"
