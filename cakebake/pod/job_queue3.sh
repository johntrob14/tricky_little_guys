#!/bin/bash
cd /workspace/cakebake; . pod/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while tmux has-session -t job-q2 2>/dev/null || tmux has-session -t job-bau1 2>/dev/null; do sleep 30; done
python -m cb.judge_check > judge_check.log 2>&1; tail -3 judge_check.log
python -m cb.train --mode rl --run rl1 --steps 60 --bs 256 --group 16 --save-every 10 --lr 5e-6 --beta 0.05 > runs_rl1.log 2>&1
grep -E '"step": 60|Traceback' runs_rl1.log
bash pod/gen_sweep.sh rl1@step_10 rl1@step_20 rl1@step_30 rl1@step_40 rl1@step_50 rl1@step_60
echo QUEUE3_DONE
