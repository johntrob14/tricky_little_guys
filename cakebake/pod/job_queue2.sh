#!/bin/bash
cd /workspace/cakebake; . pod/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m cb.judge_check 2>&1 | grep -E "qwen3-8b|Error|Traceback" -A2 | head -8
python -m cb.train --mode rl --run smoke_rl --steps 2 --bs 32 --group 8 --save-every 2 --lr 5e-6 2>&1 | grep -E '"step"|Traceback|Error' -A2 | head -10
rm -f runs_cd1.log; rm -rf runs/cd1
python -m cb.train --mode cd --run cd1 --steps 500 --bs 16 --gen-chunk 64 --save-every 50 --lr 1e-5 > runs_cd1.log 2>&1
grep -E '"step": 500|Traceback' runs_cd1.log
bash pod/gen_sweep.sh opsd_strong@step_100 opsd_strong@step_200 opsd_strong@step_300 opsd_strong@step_400 opsd_strong@step_500 cd1@step_100 cd1@step_200 cd1@step_300 cd1@step_400 cd1@step_500
echo QUEUE2_DONE
