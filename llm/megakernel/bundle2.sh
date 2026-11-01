#!/bin/bash
# one gpu-lock hold: final S and F 4000-step runs with the 85/17/6 partition (decode live)
cd ~/llm/megakernel; D=~/data
COMMON="--data $D/tinystories_cl100k_train.u32 --starts $D/starts_4000.u32 --infer-sms 17 --doom-sms 6"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== S 4000 steps"
./triad_perf2 --steps 4000 $COMMON --init $D/init_flat_S.f32 --loss-out /tmp/lossS4000_v2.txt --text-out /tmp/tokS4000_v2.txt 2>&1 | grep -E "steady|SM utilisation|decode:|wall|total|final"
tail -2 /tmp/lossS4000_v2.txt
echo "=== F 4000 steps"
./triad_perfF2 --steps 4000 $COMMON --init $D/init_flat_F.f32 --loss-out /tmp/lossF4000.txt --text-out /tmp/tokF4000.txt 2>&1 | grep -E "steady|SM utilisation|decode:|wall|total|final"
tail -2 /tmp/lossF4000.txt
