#!/bin/bash
# After opsd1: vLLM jobs alone, then CD and OPSD-strong concurrently (HF only), then the RL judge check.
cd /workspace/cakebake; . pod/env.sh
while tmux has-session -t job-opsd1 2>/dev/null; do sleep 30; done
bash pod/gen_all.sh posthoc_unmixed_dpo
python -m cb.judge_check 2>&1 | grep -E "qwen3-8b|Traceback|Error" | head -3
python -m cb.train --mode cd --run cd1 --steps 500 --bs 16 --gen-chunk 64 --save-every 50 --lr 1e-5 > runs_cd1.log 2>&1 &
python -m cb.train --mode opsd --run opsd_strong --teacher-system data/teacher_strong_system.txt --steps 500 --bs 16 --gen-chunk 64 --save-every 50 --lr 1e-5 > runs_opsd_strong.log 2>&1 &
wait
grep -E '"step": 500|Traceback' runs_cd1.log runs_opsd_strong.log
echo JOB_DONE
