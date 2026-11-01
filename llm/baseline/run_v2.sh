#!/usr/bin/env bash
# BASELINE v2 driver (TE-based Megatron, config S and F, launch counts). Each job is wrapped in the shared GPU lock. Env: REPS(3) STEPS(4000)
# TOTAL_SMS/TRAIN_SMS/INFER_SMS (A100 default 108/85/17; B200: 148/116/24), PHASES.
# PHASES: count train_te train_pt_f train_graph serve  (default all; train_te/train_pt_f/train_graph also do the SM-restricted MPS runs)
set -u
export TOTAL_SMS=${TOTAL_SMS:-108} TRAIN_SMS=${TRAIN_SMS:-85} INFER_SMS=${INFER_SMS:-17}
REPS=${REPS:-3}; STEPS=${STEPS:-4000}; PHASES=${PHASES:-"count train_te train_pt_f train_graph serve"}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; export PATH=$HOME/venv-baseline/bin:$PATH
pct() { echo $(( (100 * $1 + TOTAL_SMS / 2) / TOTAL_SMS )); }
TP=$(pct $TRAIN_SMS); IP=$(pct $INFER_SMS); LK="flock -w 14400 $HOME/.gpu.lock"
has() { [[ " $PHASES " == *" $1 "* ]]; }
run() { $LK ./one_run.sh "$@"; }
export VLLM_ENABLE_V1_MULTIPROCESSING=0
r() { $LK ./one_run.sh "$1" "$2" bash -c "$3"; }   # r NAME PCT 'cmd using $OUT'
CKF=$HOME/data/ckpt_pytorch_F_step$STEPS.pt
if has count; then
  for m in S F; do
    r c3_megatron_te_$m 0 "MODEL=$m python megatron_train.py --impl te --count3 --out \$OUT"
    r c3_pytorch_$m 0 "MODEL=$m python train_pytorch.py --count3 --out \$OUT"
  done
fi
if has train_te; then for m in S F; do for i in $(seq $REPS); do
  r megatron_te_${m}_full_r$i 0 "MODEL=$m python megatron_train.py --impl te --steps $STEPS --out \$OUT --tag full"
  r megatron_te_${m}_mps_r$i $TP "MODEL=$m python megatron_train.py --impl te --steps $STEPS --out \$OUT --tag mps$TRAIN_SMS"
done; done; fi
if has train_pt_f; then for i in $(seq $REPS); do
  X=""; [ $i = 1 ] && X="--save $CKF"
  r pytorch_F_full_r$i 0 "MODEL=F python train_pytorch.py --steps $STEPS --out \$OUT --tag full $X"
  r pytorch_F_mps_r$i $TP "MODEL=F python train_pytorch.py --steps $STEPS --out \$OUT --tag mps$TRAIN_SMS"
done; fi
if has train_graph; then for m in S F; do for i in $(seq $REPS); do
  r pytorch_graph_${m}_full_r$i 0 "MODEL=$m python train_pytorch.py --cuda_graph --steps $STEPS --out \$OUT --tag full"
  r megatron_te_graph_${m}_full_r$i 0 "MODEL=$m python megatron_train.py --impl te --cuda_graph --steps $STEPS --out \$OUT --tag full"
done; done; fi
if has serve; then
  HFS=$HOME/data/hf_trained; [ -d $HFS ] || python make_hf.py $HFS $HOME/data/ckpt_pytorch_step$STEPS.pt >/dev/null
  HFF=$HOME/data/hf_trained_F; [ -d $HFF ] || { [ -f $CKF ] && MODEL=F python make_hf.py $HFF $CKF >/dev/null || { HFF=$HOME/data/hf_init_F; MODEL=F python make_hf.py $HFF >/dev/null; }; }
  echo "serving weights S=$HFS F=$HFF" | tee results/runs_v2/serve_weights.txt
  for m in S F; do HF=$HFS; [ $m = F ] && HF=$HFF
    for i in $(seq $REPS); do
      r vllm_${m}_full_r$i 0 "MODEL=$m python serve_vllm.py $HF --reps 1 --out \$OUT"
      r vllm_${m}_mps_r$i $IP "MODEL=$m python serve_vllm.py $HF --reps 1 --out \$OUT"
    done
    for cg in default eager full; do r vllmc3_${m}_$cg 0 "MODEL=$m python serve_vllm.py $HF --reps 1 --count_launches --cg $cg --out \$OUT"; done
  done
fi
echo ALL_DONE
