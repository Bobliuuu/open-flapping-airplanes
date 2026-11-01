#!/usr/bin/env bash
# run_all.sh : B200 baseline calibration driver (FA's method). UNTESTED ON B200 (adapted from llm/baseline/run_v2.sh).
# Every job runs ALONE on the GPU, one after the other (FA times Megatron and vLLM individually, no Doom):
#   Megatron-core + Transformer Engine (+flash-attn/cuDNN attention), models S and F, 4000 steps, full GPU and restricted to 116 SMs via MPS
#   (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=78); PyTorch eager as a second reference; vLLM batch-1 decode full GPU and on 24 SMs (16%).
#   Launch counts (torch.profiler/CUPTI, 3 definitions) as in llm/baseline; optional nsys cross-check: ./count_nsys.sh
# Output: results/runs_v2/*.json (raw) and results/baseline_v2_b200_<date>.json (aggregate_v2.py; SAME schema as llm/baseline/results/baseline_v2_a100_*.json).
#
# Env: PHASES  (default "sms count train_te train_pt train_graph serve")   REPS (3)   STEPS (4000)
#      TOTAL_SMS/TRAIN_SMS/INFER_SMS (148/116/24)   TRAIN_PCT (78)   INFER_PCT (16)   ATTN (auto|flash|fused: Transformer Engine attention backend)
#      PY_TRAIN / PY_SERVE (interpreters from setup_b200.sh)   GPU_LOCK (optional flock file, e.g. ~/.gpu.lock)   GPU_TAG (b200)
# Quick smoke test first:  STEPS=200 REPS=1 PHASES="count train_te" ./run_all.sh
set -u
export TOTAL_SMS=${TOTAL_SMS:-148} TRAIN_SMS=${TRAIN_SMS:-116} INFER_SMS=${INFER_SMS:-24}
REPS=${REPS:-3}; STEPS=${STEPS:-4000}; PHASES=${PHASES:-"sms count train_te train_pt train_graph serve"}
TRAIN_PCT=${TRAIN_PCT:-78}; INFER_PCT=${INFER_PCT:-16}; export GPU_TAG=${GPU_TAG:-b200}
PY_TRAIN=${PY_TRAIN:-$HOME/venv-train/bin/python}; PY_SERVE=${PY_SERVE:-$HOME/venv-serve/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; mkdir -p results/runs_v2
has() { [[ " $PHASES " == *" $1 "* ]]; }
LK=""; [ -n "${GPU_LOCK:-}" ] && LK="flock -w 14400 $GPU_LOCK"
case "${ATTN:-auto}" in
  flash) export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 ;;
  fused) export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 ;;
  auto)  : ;;   # TE picks (prefers cuDNN fused on Blackwell, flash-attn if cuDNN rejects); the chosen backend is logged with NVTE_DEBUG=1
esac
export NVTE_DEBUG=${NVTE_DEBUG:-1} NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-1}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_DEVICE_MAX_CONNECTIONS=1
r() { $LK ./one_run.sh "$1" "$2" bash -c "$3"; }   # r NAME MPS_PCT 'cmd using $OUT'   (MPS_PCT 0 = full GPU)
T="$PY_TRAIN"; S="$PY_SERVE"
{ echo "date: $(date -Is)"; nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv; $T -c "import torch;print('torch',torch.__version__,torch.version.cuda)";
  $T -c "import te_env,transformer_engine as t,transformer_engine.pytorch" 2>&1 | tail -1; $T -m pip list 2>/dev/null | grep -i -E "^(torch|transformer.engine|flash.attn|megatron)"; $S -m pip list 2>/dev/null | grep -i -E "^(vllm|torch) "; env | grep -E "^NVTE_|^TRAIN|^INFER|^TOTAL"; } > results/runs_v2/environment.txt 2>&1

if has sms; then ./calibrate_sms.sh "$TRAIN_PCT" "$INFER_PCT" 100 2>&1 | tee results/runs_v2/sms_check.txt; fi

if has count; then   # launch counts: (a) kernels executed, (b) host launch calls, (c) host launch calls with the unit replayed from one CUDA graph
  for m in S F; do
    r c3_megatron_te_$m 0 "MODEL=$m $T megatron_train.py --impl te --count3 --out \$OUT"
    r c3_pytorch_$m 0 "MODEL=$m $T train_pytorch.py --count3 --out \$OUT"
  done
fi
if has train_te; then for m in S F; do for i in $(seq $REPS); do
  r megatron_te_${m}_full_r$i 0 "MODEL=$m $T megatron_train.py --impl te --steps $STEPS --out \$OUT --tag full"
  r megatron_te_${m}_mps_r$i $TRAIN_PCT "MODEL=$m $T megatron_train.py --impl te --steps $STEPS --out \$OUT --tag mps$TRAIN_SMS"
done; done; fi
CKS=$HOME/data/ckpt_pytorch_step$STEPS.pt; CKF=$HOME/data/ckpt_pytorch_F_step$STEPS.pt
if has train_pt; then for m in S F; do for i in $(seq $REPS); do
  X=""; CK=$CKS; [ $m = F ] && CK=$CKF; [ $i = 1 ] && X="--save $CK"
  r pytorch_${m}_full_r$i 0 "MODEL=$m $T train_pytorch.py --steps $STEPS --out \$OUT --tag full $X"
  r pytorch_${m}_mps_r$i $TRAIN_PCT "MODEL=$m $T train_pytorch.py --steps $STEPS --out \$OUT --tag mps$TRAIN_SMS"
done; done; fi
if has train_graph; then for m in S F; do for i in $(seq $REPS); do
  r pytorch_graph_${m}_full_r$i 0 "MODEL=$m $T train_pytorch.py --cuda_graph --steps $STEPS --out \$OUT --tag full"
  r megatron_te_graph_${m}_full_r$i 0 "MODEL=$m $T megatron_train.py --impl te --cuda_graph --steps $STEPS --out \$OUT --tag full"
done; done; fi
if has serve; then
  HFS=$HOME/data/hf_trained; HFF=$HOME/data/hf_trained_F
  # make_hf.py needs torch+transformers: the train venv has both (transformers installed by setup_b200.sh)
  [ -d $HFS ] || { [ -f $CKS ] && MODEL=S $T make_hf.py $HFS $CKS >/dev/null || { HFS=$HOME/data/hf_init; MODEL=S $T make_hf.py $HFS >/dev/null; }; }
  [ -d $HFF ] || { [ -f $CKF ] && MODEL=F $T make_hf.py $HFF $CKF >/dev/null || { HFF=$HOME/data/hf_init_F; MODEL=F $T make_hf.py $HFF >/dev/null; }; }
  echo "serving weights S=$HFS F=$HFF" | tee results/runs_v2/serve_weights.txt
  for m in S F; do HF=$HFS; [ $m = F ] && HF=$HFF
    for i in $(seq $REPS); do
      r vllm_${m}_full_r$i 0 "MODEL=$m $S serve_vllm.py $HF --reps 1 --out \$OUT"
      r vllm_${m}_mps_r$i $INFER_PCT "MODEL=$m $S serve_vllm.py $HF --reps 1 --out \$OUT"
    done
    for cg in default eager full; do r vllmc3_${m}_$cg 0 "MODEL=$m $S serve_vllm.py $HF --reps 1 --count_launches --cg $cg --out \$OUT"; done
  done
fi
# aggregate_v2.py imports common.py (torch); use the train interpreter
$T aggregate_v2.py && $T fa_compare.py
echo ALL_DONE
