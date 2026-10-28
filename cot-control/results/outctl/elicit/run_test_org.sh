#!/bin/bash
# Held-out test on CoT-control organisms: baseline (1 sample) and the parent model's dev-chosen variant (2 samples); no dev tuning per organism.
# Two process chains at a time (the box OOMs with many).
cd /home/john/code/tlg-inksm/cot-control/rl; set -a; . ~/.config/sauce/tinker.env; set +a
run() { timeout 6000 uv run python ../results/outctl/elicit/elicit.py $1 base test 1 2>&1 | grep -E "done|FAILED|Error|cost"
        timeout 6000 uv run python ../results/outctl/elicit/elicit.py $1 $2 test 2 2>&1 | grep -E "done|FAILED|Error|cost"; }
( run mt1b shotstack; run grpo3 shotstack; run opsdgrpo shotlow ) &
( run mt2e shotstack; run gsft1 shotstack ) &
wait
uv run python ../results/outctl/elicit/grade.py 2>&1 | grep -c graded
cd ../results/outctl; python3 elicit/summary.py test
