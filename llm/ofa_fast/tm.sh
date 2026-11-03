#!/bin/bash
# usage: tm.sh STEPS bin1 bin2 ...   (runs under one lock hold)
cd ~/llm/ofa_fast; ST=$1; shift
A="--steps $ST --infer-sms 17 --doom-sms 6 --data $HOME/data/tinystories_cl100k_train.u32 --init $HOME/data/init_flat_S.f32 --starts $HOME/data/starts_4000.u32 --prompt $HOME/data/prompt_ids.txt --wad $HOME/doom-gpu/doom1.wad --vocab $HOME/data/cl100k_vocab.bin --port 8099 --exit"
for b in "$@"; do echo "=== $b"; timeout 600 ./$b $A $EXTRA 2>&1 | grep -E "steady state|^loss|decode:|doom:|error|Error"; done
