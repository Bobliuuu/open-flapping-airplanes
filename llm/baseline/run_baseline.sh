#!/usr/bin/env bash
# Reproduce the whole baseline. Run on the GPU box (A100: defaults; B200: TOTAL_SMS=148 TRAIN_SMS=116 INFER_SMS=24 ./run_baseline.sh).
# Needs ~/venv-baseline (see README.md) and ~/data/tinystories_cl100k_{train,valid}.u32 (python prep_data.py).
# Env: CUDA_ARCH(sm_80; sm_100 on B200) TOTAL_SMS TRAIN_SMS INFER_SMS REPS(3) STEPS(4000) PHASES("golden train train_mps serve serve_mps text") IDLE_WAIT_S(600)
set -u
export TOTAL_SMS=${TOTAL_SMS:-108} TRAIN_SMS=${TRAIN_SMS:-85} INFER_SMS=${INFER_SMS:-17}
REPS=${REPS:-3}; STEPS=${STEPS:-4000}; IDLE_WAIT_S=${IDLE_WAIT_S:-600}
PHASES=${PHASES:-"golden train train_mps serve serve_mps text"}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export PATH=$HOME/venv-baseline/bin:$PATH
R=$HERE/results/runs; mkdir -p "$R"
pct() { echo $(( (100 * $1 + TOTAL_SMS / 2) / TOTAL_SMS )); }
TRAIN_PCT=$(pct $TRAIN_SMS); INFER_PCT=$(pct $INFER_SMS)
MPSDIR=$HOME/mps_baseline
mps_start() { export CUDA_MPS_PIPE_DIRECTORY=$MPSDIR/pipe CUDA_MPS_LOG_DIRECTORY=$MPSDIR/log; mkdir -p $MPSDIR/pipe $MPSDIR/log
  nvidia-cuda-mps-control -d; sleep 2; }
mps_stop() { [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] && { echo quit | nvidia-cuda-mps-control; sleep 2; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; }; }
trap mps_stop EXIT
# wait until GPU util < 5%; after IDLE_WAIT_S proceed anyway and flag the run as noisy
wait_idle() { local t0=$(date +%s) u
  while :; do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    [ "$u" -lt 5 ] && { echo 0; return; }
    [ $(( $(date +%s) - t0 )) -gt $IDLE_WAIT_S ] && { echo 1; return; }
    sleep 3; done; }
# timed NAME cmd...  (cmd must accept --out via the env var OUT)
timed() { local name=$1; shift; local noisy; noisy=$(wait_idle)
  local u0=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)
  echo "== $name noisy=$noisy gpu_before=[$u0]"
  ( while :; do nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; sleep 2; done > $R/$name.procs ) & local sp=$!
  OUT=$R/$name.json "$@" > $R/$name.log 2>&1; local rc=$?; kill $sp 2>/dev/null; wait $sp 2>/dev/null
  local maxp=$(sort -n $R/$name.procs | tail -1)
  local u1=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)
  echo "{\"name\":\"$name\",\"noisy_not_idle_before_start\":$noisy,\"gpu_before\":\"$u0\",\"gpu_after\":\"$u1\",\"max_compute_procs_seen\":${maxp:-0},\"rc\":$rc}" > $R/$name.meta.json
  tail -1 $R/$name.log | cut -c1-300; }
has() { [[ " $PHASES " == *" $1 "* ]]; }
CK=$HOME/data/ckpt_pytorch_step${STEPS}.pt

has golden && python train_pytorch.py --golden > $R/golden.log 2>&1
if has train; then
  for i in $(seq $REPS); do
    EXTRA=""; [ $i = 1 ] && EXTRA="--count_launches --save $CK"
    timed pytorch_full_r$i bash -c "python train_pytorch.py --steps $STEPS --out \$OUT --tag full $EXTRA"
    EXTRA=""; [ $i = 1 ] && EXTRA="--count_launches"
    timed megatron_full_r$i bash -c "python megatron_train.py --steps $STEPS --out \$OUT --tag full $EXTRA"
  done
fi
if has train_mps; then
  mps_start; echo "SM check @${TRAIN_PCT}%: $(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$TRAIN_PCT ./baseline_check_sms)" | tee $R/sms_train.txt
  for i in $(seq $REPS); do
    timed pytorch_mps_r$i bash -c "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$TRAIN_PCT python train_pytorch.py --steps $STEPS --out \$OUT --tag mps$TRAIN_SMS"
    timed megatron_mps_r$i bash -c "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$TRAIN_PCT python megatron_train.py --steps $STEPS --out \$OUT --tag mps$TRAIN_SMS"
  done; mps_stop
fi
HF=$HOME/data/hf_trained; [ -f "$CK" ] && python make_hf.py $HF $CK >/dev/null 2>&1 || { HF=$HOME/data/hf_init; python make_hf.py $HF >/dev/null 2>&1; }
echo "serving weights: $HF" | tee $R/serve_weights.txt
export VLLM_ENABLE_V1_MULTIPROCESSING=0
if has serve; then for i in $(seq $REPS); do
  EXTRA=""; [ $i = 1 ] && EXTRA="--count_launches"
  timed vllm_full_r$i bash -c "python serve_vllm.py $HF --reps 1 --out \$OUT $EXTRA"; done; fi
if has serve_mps; then
  mps_start; echo "SM check @${INFER_PCT}%: $(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$INFER_PCT ./baseline_check_sms)" | tee $R/sms_infer.txt
  for i in $(seq $REPS); do
    timed vllm_mps_r$i bash -c "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$INFER_PCT python serve_vllm.py $HF --reps 1 --out \$OUT"; done; mps_stop
fi
has text && OUT=$R/text.json python serve_vllm.py $HF --text --out $R/text.json > $R/text.log 2>&1
python aggregate.py
