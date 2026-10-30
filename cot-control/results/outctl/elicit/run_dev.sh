#!/bin/bash
# 7 prompt variants x 5 models on the dev split (1 sample), then grade and summarize. Resumable: elicit.py skips finished items.
cd /home/john/code/tlg-arena-rl/cot-control/rl; set -a; . ~/.config/sauce/tinker.env; set +a
for m in gptoss qwen38 qwen35 ds41 glm53; do
  timeout 6000 uv run python ../results/outctl/elicit/elicit.py $m think,sys,before,rep,selfmon,stack,shot dev 1 2>&1 | grep -E "done|FAILED|Error" &
done; wait
uv run python ../results/outctl/elicit/grade.py 2>&1 | grep -c graded
cd ../results/outctl; python3 elicit/summary.py dev -v
