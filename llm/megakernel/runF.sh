#!/usr/bin/env bash
# usage: ./runS.sh <binary> <steps> [extra args]   (model F, under the shared GPU lock)
B=$1; S=$2; shift 2
ssh ubuntu@<ip> "cd ~/llm/megakernel && flock -w 7200 ~/.gpu.lock sh -c 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader; ./$B --steps $S --data ~/data/tinystories_cl100k_train.u32 --init ~/data/init_flat_F.f32 --starts ~/data/starts_4000.u32 $*' 2>&1"
