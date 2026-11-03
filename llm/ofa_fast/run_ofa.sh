#!/usr/bin/env bash
# Run the open-flapping-airplanes LLM kernel (training + live text + real Doom, one launch) and watch it in your browser. From your Mac:
#   [MODEL=S|F] ./llm/ofa/run_ofa.sh ubuntu@<ip> [port=8080] [extra ofa args, e.g. --steps 4000]
#   MODEL=S (default): 12 layers, d=768, 162M params.  MODEL=F: 4 layers, d=2048, ~408M params (the Flapping-Airplanes-shaped config; builds with -DMODEL_F).
# One-time box setup first: ./doom-gpu/setup_box.sh ubuntu@<ip>, plus the data/weights from llm/baseline (~/data/*.u32, init_flat_$MODEL.f32,
# cl100k_vocab.bin, prompt_ids.txt via llm/ofa/make_vocab.py). Takes the GPU lock, so it waits if another job holds the A100.
set -euo pipefail
HOST="${1:?usage: run_ofa.sh user@host [port] [ofa args]}"; PORT="${2:-8080}"; shift || true; shift || true
MODEL="${MODEL:-S}"; case "$MODEL" in S) MFLAG="";; F) MFLAG="-DMODEL_F";; *) echo "MODEL must be S or F" >&2; exit 2;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$HERE/../.."
rsync -az --exclude '*.o' --exclude ofa --exclude '*.log' "$ROOT/llm/gemm" "$ROOT/llm/attention" "$ROOT/llm/elementwise" "$ROOT/llm/decode" "$HOST:~/llm/"
rsync -az --exclude ofa --exclude '*.log' --exclude '*.png' "$HERE/" "$HOST:~/llm/ofa_fast/"
rsync -az --exclude doomgeneric --exclude obj --exclude src_gpu --exclude '*.o' --exclude '*.ptx' --exclude '*.bc' --exclude '*.ll' --exclude '*.cubin' "$ROOT/doom-gpu/" "$HOST:~/doom-gpu/"
( until curl -s "http://localhost:$PORT/state" >/dev/null 2>&1; do sleep 1; done; sleep 1; open "http://localhost:$PORT" ) &
ssh -t -L "$PORT:localhost:$PORT" "$HOST" "cd ~/doom-gpu && { [ -f doom.ptx ] || ./build_doom.sh >/dev/null; } && cd ~/llm/ofa_fast && \
  [ -f \$HOME/data/cl100k_vocab.bin ] || ~/venv-baseline/bin/python make_vocab.py \$HOME/data/cl100k_vocab.bin \$HOME/data/prompt_ids.txt 'Once upon a time'; \
  if ! ./build_merged.sh ofa_fast_$MODEL -DMERGED_DOOM $MFLAG; then echo 'build_merged.sh FAILED (nvcc dryrun layout or ptxmerge step changed?); nvcc version:' >&2; nvcc --version | tail -2 >&2; exit 1; fi && \
  [ -x ofa_fast_$MODEL ] || { echo 'merged binary missing; nvcc version:' >&2; nvcc --version | tail -2 >&2; exit 1; } && \
  flock -w 7200 \$HOME/.gpu.lock ./ofa_fast_$MODEL --steps 4000 --infer-sms 17 --doom-sms 6 --port $PORT --wad \$HOME/doom-gpu/doom1.wad --vocab \$HOME/data/cl100k_vocab.bin \
    --prompt \$HOME/data/prompt_ids.txt --prompt-text 'Once upon a time' --data \$HOME/data/tinystories_cl100k_train.u32 --init \$HOME/data/init_flat_$MODEL.f32 --starts \$HOME/data/starts_4000.u32 $*"
