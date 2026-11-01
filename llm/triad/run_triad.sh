#!/usr/bin/env bash
# Run the Triad LLM kernel (training + live text + real Doom, one launch) and watch it in your browser. From your Mac:
#   [MODEL=S|F] ./llm/triad/run_triad.sh ubuntu@<ip> [port=8080] [extra triad args, e.g. --steps 4000]
#   MODEL=S (default): 12 layers, d=768, 162M params.  MODEL=F: 4 layers, d=2048, ~408M params (the Flapping-Airplanes-shaped config; builds with -DMODEL_F).
# One-time box setup first: ./doom-gpu/setup_box.sh ubuntu@<ip>, plus the data/weights from llm/baseline (~/data/*.u32, init_flat_$MODEL.f32,
# cl100k_vocab.bin, prompt_ids.txt via llm/triad/make_vocab.py). Takes the GPU lock, so it waits if another job holds the A100.
set -euo pipefail
HOST="${1:?usage: run_triad.sh user@host [port] [triad args]}"; PORT="${2:-8080}"; shift || true; shift || true
MODEL="${MODEL:-S}"; case "$MODEL" in S) MFLAG="";; F) MFLAG="-DMODEL_F";; *) echo "MODEL must be S or F" >&2; exit 2;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$HERE/../.."
rsync -az --exclude '*.o' --exclude triad --exclude '*.log' "$ROOT/llm/gemm" "$ROOT/llm/attention" "$ROOT/llm/elementwise" "$ROOT/llm/decode" "$HOST:~/llm/"
rsync -az --exclude triad --exclude '*.log' --exclude '*.png' "$HERE/" "$HOST:~/llm/triad/"
rsync -az --exclude doomgeneric --exclude obj --exclude src_gpu --exclude '*.o' --exclude '*.ptx' --exclude '*.bc' --exclude '*.ll' --exclude '*.cubin' "$ROOT/doom-gpu/" "$HOST:~/doom-gpu/"
( until curl -s "http://localhost:$PORT/state" >/dev/null 2>&1; do sleep 1; done; sleep 1; open "http://localhost:$PORT" ) &
ssh -t -L "$PORT:localhost:$PORT" "$HOST" "cd ~/doom-gpu && { [ -f doom_dev.o ] || ./build_doom.sh >/dev/null; } && cd ~/llm/triad && \
  [ -f \$HOME/data/cl100k_vocab.bin ] || ~/venv-baseline/bin/python make_vocab.py \$HOME/data/cl100k_vocab.bin \$HOME/data/prompt_ids.txt 'Once upon a time'; \
  nvcc -arch=sm_80 -O3 -std=c++17 --extended-lambda -rdc=true -maxrregcount=128 $MFLAG triad.cu \$HOME/doom-gpu/doom_dev.o -o triad -lpthread && \
  flock -w 7200 \$HOME/.gpu.lock ./triad --steps 4000 --infer-sms 17 --doom-sms 6 --port $PORT --wad \$HOME/doom-gpu/doom1.wad --vocab \$HOME/data/cl100k_vocab.bin \
    --prompt \$HOME/data/prompt_ids.txt --prompt-text 'Once upon a time' --data \$HOME/data/tinystories_cl100k_train.u32 --init \$HOME/data/init_flat_$MODEL.f32 --starts \$HOME/data/starts_4000.u32 $*"
