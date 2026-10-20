#!/bin/bash
cd /workspace/cakebake; . pod/env.sh; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while tmux has-session -t job-q4b 2>/dev/null; do sleep 30; done
# RL warm-started from the early OPSD checkpoint (QER ~17%); KL anchor = that checkpoint (frozen copy of --model)
python -m cb.train --mode rl --run rl4 --model runs/opsd1/step_100 --steps 30 --bs 256 --group 16 --save-every 10 --lr 5e-6 --beta 0.05 --judge-thinking > runs_rl4.log 2>&1
grep -E '"step": 30,|Traceback' runs_rl4.log | head -3
bash pod/gen_sweep.sh rl4@step_10 rl4@step_20 rl4@step_30 2>&1 | grep -E "GEN_DONE|GEN_FAILED"
python -m cb.probe --model runs/rl4/step_30 --name rl4_step_30 --gpu-mem 0.5 2>&1 | grep -E "PROBE_DONE|Error:" | head -2
echo Q5_DONE
