#!/usr/bin/env bash
# run_b200.sh : run the open-flapping-airplanes megakernel on the local GPU (run it ON the B200 box).  UNTESTED on B200.
#   [MODEL=S|F] ./run_b200.sh <binary> [extra ofa args]      e.g.  ./run_b200.sh ./ofa_b200_sm_100a --steps 4000
# Partition: --split auto (116/24/8 on a 148-SM B200, 85/17/6 on a 108-SM A100). Override with --infer-sms/--doom-sms/--train-sms.
# Data: see llm/baseline (tinystories_cl100k_train.u32, init_flat_$MODEL.f32, starts_4000.u32) and llm/ofa/make_vocab.py (cl100k_vocab.bin, prompt_ids.txt).
set -euo pipefail
BIN=${1:?binary}; shift || true
MODEL=${MODEL:-S}; D=${DATA_DIR:-$HOME/data}
LK=""; [ -n "${GPU_LOCK:-}" ] && LK="flock -w 7200 $GPU_LOCK"
WAD=""; [ -f "$HOME/doom-gpu/doom1.wad" ] && WAD="--wad $HOME/doom-gpu/doom1.wad"
cd "$(dirname "$0")"
$LK "$BIN" --split auto $WAD --vocab $D/cl100k_vocab.bin --prompt $D/prompt_ids.txt --prompt-text 'Once upon a time' \
  --data $D/tinystories_cl100k_train.u32 --init $D/init_flat_$MODEL.f32 --starts $D/starts_4000.u32 "$@"
