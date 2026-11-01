#!/usr/bin/env bash
# one_run.sh NAME MPS_PCT cmd...   (call as: flock -w 7200 ~/.gpu.lock ./one_run.sh ...). Runs ONE timed job on the (idle) GPU; MPS_PCT=0 -> full GPU.
# Command sees $OUT (result json path). Writes results/runs_v2/NAME.{json,log,meta.json}.
set -u
name=$1; pct=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd); R=$HERE/results/runs_v2; mkdir -p $R; export PATH=$HOME/venv-baseline/bin:$PATH
MPSDIR=$HOME/mps_baseline
t0=$(date +%s); noisy=0
while :; do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  [ "$u" -lt 5 ] && break; [ $(( $(date +%s) - t0 )) -gt ${IDLE_WAIT_S:-300} ] && { noisy=1; break; }; sleep 3; done
if [ "$pct" != 0 ]; then export CUDA_MPS_PIPE_DIRECTORY=$MPSDIR/pipe CUDA_MPS_LOG_DIRECTORY=$MPSDIR/log; mkdir -p $MPSDIR/pipe $MPSDIR/log
  nvidia-cuda-mps-control -d; sleep 2; export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct; fi
u0=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)
( while :; do nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; sleep 2; done > $R/$name.procs ) & sp=$!
OUT=$R/$name.json "$@" > $R/$name.log 2>&1; rc=$?
kill $sp 2>/dev/null; wait $sp 2>/dev/null
if [ "$pct" != 0 ]; then echo quit | nvidia-cuda-mps-control; sleep 2; fi
maxp=$(sort -n $R/$name.procs | tail -1); u1=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)
echo "{\"name\":\"$name\",\"mps_pct\":$pct,\"noisy_not_idle_before_start\":$noisy,\"gpu_before\":\"$u0\",\"gpu_after\":\"$u1\",\"max_compute_procs_seen\":${maxp:-0},\"rc\":$rc}" > $R/$name.meta.json
echo "$name rc=$rc noisy=$noisy maxp=$maxp: $(tail -1 $R/$name.log | cut -c1-200)"
