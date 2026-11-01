#!/bin/bash
# runs on the box, under ONE gpu-lock hold: S regression + config F validation + F timing
cd ~/llm/megakernel; D=~/data
COMMON="--data $D/tinystories_cl100k_train.u32 --starts $D/starts_4000.u32"
PART="--infer-sms 17 --doom-sms 6"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== S: new (grouped AdamW) vs old (fetch-add) 60 steps with decode"
./ofa_perf2 --steps 60 $COMMON --init $D/init_flat_S.f32 $PART --loss-out /tmp/lossS60_new.txt 2>&1 | grep -E "ms/step|train|decode|tok/s|SM" | tail -8
./ofa_perf1 --steps 60 $COMMON --init $D/init_flat_S.f32 $PART --loss-out /tmp/lossS60_old.txt 2>&1 | grep -E "ms/step" | tail -2
diff /tmp/lossS60_new.txt /tmp/lossS60_old.txt && echo "S LOSSES IDENTICAL"
echo "=== F: step-0 grads (no-opt)"
./ofa_perfF2 --steps 1 --no-opt $COMMON --init $D/init_flat_F.f32 --dump-grads /tmp/gF.f32 --loss-out /tmp/lossF0.txt 2>&1 | grep -E "memory|graph|wrote|error|Error" 
cat /tmp/lossF0.txt
echo "=== F: one step with AdamW"
./ofa_perfF2 --steps 1 $COMMON --init $D/init_flat_F.f32 --dump-w /tmp/wF1.f32 --loss-out /tmp/lossF1.txt 2>&1 | grep -E "wrote|error|Error"
echo "=== F: 100 steps"
./ofa_perfF2 --steps 100 $COMMON --init $D/init_flat_F.f32 --loss-out /tmp/lossF100.txt 2>&1 | grep -E "ms/step|error|Error" | tail -3
echo "=== F: pytorch reference 100 steps"
MODEL=F ~/venv-baseline/bin/python ref_F.py 100 /tmp/refF100.json /tmp/refF_w1.f32 2>&1 | tail -3
echo "=== F timing with decode partition, prof step 50"
./ofa_perfF2 --steps 60 $COMMON --init $D/init_flat_F.f32 $PART --prof-step 50 --loss-out /tmp/lossF60p.txt 2>&1 | tail -60
