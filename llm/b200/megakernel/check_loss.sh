#!/usr/bin/env bash
# check_loss.sh : 20-step loss regression of the parameterised build against a reference binary built from the ORIGINAL llm/ofa/ofa.cu.
#   ./check_loss.sh <new_binary> <reference_binary> [steps=20]      (run under flock on a shared box; short GPU run)
# Both run with identical explicit flags (training only, no decode/Doom) and write per-step losses with --loss-out; the files must match exactly.
set -u
NEW=$1; REF=$2; STEPS=${3:-20}; D=${DATA_DIR:-$HOME/data}; MODEL=${MODEL:-S}
A="--steps $STEPS --data $D/tinystories_cl100k_train.u32 --init $D/init_flat_$MODEL.f32 --starts $D/starts_4000.u32"
$REF $A --loss-out /tmp/loss_ref.txt > /tmp/ref.out 2>&1; $NEW $A --loss-out /tmp/loss_new.txt > /tmp/new.out 2>&1
tail -3 /tmp/ref.out | cut -c1-160; tail -3 /tmp/new.out | cut -c1-160
if cmp -s /tmp/loss_ref.txt /tmp/loss_new.txt; then echo "LOSS IDENTICAL ($STEPS steps, $(wc -l < /tmp/loss_new.txt) lines): $(tail -1 /tmp/loss_new.txt)"; else echo "LOSS DIFFERS"; diff /tmp/loss_ref.txt /tmp/loss_new.txt | head; fi
