#!/bin/bash
# Round 2 on dev: shot4 and shotstack for every model; shotlow (4 shots + low effort) for gpt-oss and Qwen3.8. One process per model.
cd /home/john/code/tlg-arena-rl/cot-control/rl; set -a; . ~/.config/sauce/tinker.env; set +a
for mv in "gptoss shot4,shotstack,shotlow" "qwen38 shot4,shotstack,shotlow" "qwen35 shot4,shotstack" "ds41 shot4,shotstack" "glm53 shot4,shotstack"; do
  set -- $mv; timeout 6000 uv run python ../results/outctl/elicit/elicit.py $1 $2 dev 1 2>&1 | grep -E "done|FAILED|Error" &
done; wait
uv run python ../results/outctl/elicit/grade.py 2>&1 | grep -c graded
cd ../results/outctl; python3 elicit/summary.py dev
