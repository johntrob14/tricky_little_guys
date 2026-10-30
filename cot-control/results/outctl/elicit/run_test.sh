#!/bin/bash
# Held-out test (72 items): baseline (1 sample) and each model's dev-best variant (2 samples). Dev choice: highest strict, ties by partial.
cd /home/john/code/tlg-arena-rl/cot-control/rl; set -a; . ~/.config/sauce/tinker.env; set +a
for mv in "gptoss shotstack" "qwen38 shotlow" "qwen35 shotstack" "ds41 shotstack" "glm53 shotstack"; do
  set -- $mv
  ( timeout 6000 uv run python ../results/outctl/elicit/elicit.py $1 base test 1 2>&1 | grep -E "done|FAILED|Error"
    timeout 6000 uv run python ../results/outctl/elicit/elicit.py $1 $2 test 2 2>&1 | grep -E "done|FAILED|Error" ) &
done; wait
uv run python ../results/outctl/elicit/grade.py 2>&1 | grep -c graded
cd ../results/outctl; python3 elicit/summary.py test -v
