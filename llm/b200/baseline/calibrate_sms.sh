#!/usr/bin/env bash
# calibrate_sms.sh : which CUDA_MPS_ACTIVE_THREAD_PERCENTAGE really gives 116 / 24 SMs on this GPU?  UNTESTED on B200.
# MPS may round the percentage to a hardware granularity (on A100 79% -> 84 SMs, 16% -> 16). The result below is what run_all.sh should use.
# Usage: ./calibrate_sms.sh [pct list]   Output: lines "pct=NN device_reported_sms=.. distinct_smids_used=.." and the best pct for TRAIN_SMS/INFER_SMS.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
TOTAL_SMS=${TOTAL_SMS:-148}; TRAIN_SMS=${TRAIN_SMS:-116}; INFER_SMS=${INFER_SMS:-24}
ARCH=${CUDA_ARCH:-sm_100a}
[ -x ./baseline_check_sms ] || nvcc -arch=$ARCH -O2 -o baseline_check_sms check_sms.cu || { echo "nvcc build failed"; exit 1; }
MPSDIR=$HOME/mps_baseline; export CUDA_MPS_PIPE_DIRECTORY=$MPSDIR/pipe CUDA_MPS_LOG_DIRECTORY=$MPSDIR/log; mkdir -p $MPSDIR/pipe $MPSDIR/log
echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; nvidia-cuda-mps-control -d; sleep 2
LIST=${*:-"8 10 12 14 15 16 17 18 20 22 25 50 70 74 75 76 77 78 79 80 81 82 85 90 100"}
echo "full GPU (no MPS env): $(env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE ./baseline_check_sms)"
declare -A got
for p in $LIST; do r=$(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$p ./baseline_check_sms); echo "pct=$p $r"; got[$p]=$(echo "$r" | sed -n 's/.*distinct_smids_used=\([0-9]*\).*/\1/p'); done
best() { local tgt=$1 bp= bd=9999; for p in "${!got[@]}"; do d=$(( got[$p] > tgt ? got[$p]-tgt : tgt-got[$p] )); [ -n "${got[$p]}" ] && [ $d -lt $bd ] && { bd=$d; bp=$p; }; done; echo "$bp (gives ${got[$bp]:-?} SMs, target $tgt)"; }
echo "BEST pct for TRAIN_SMS=$TRAIN_SMS: $(best $TRAIN_SMS)"
echo "BEST pct for INFER_SMS=$INFER_SMS: $(best $INFER_SMS)"
echo "(FA labels: 'Megatron 116 SMs', 'vLLM 24 SMs'. Default in run_all.sh: TRAIN_PCT=78 INFER_PCT=16; override with env if the best pct differs.)"
echo quit | nvidia-cuda-mps-control >/dev/null 2>&1
