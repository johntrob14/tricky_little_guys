#!/bin/bash
# Astra (108 items, medium effort, T 1) on a gpt-oss checkpoint: run_astra_ckpt.sh NAME TINKER_PATH; same settings as the other gpt-oss Astra runs.
R=/home/john/code/tlg-arena-rl/cot-control/results/tinker-runs/astra-$1-medium-t1
M=$2
cd /home/john/code/tlg-arena-rl/cot-control/tinker-runner && set -a && . ~/.config/sauce/tinker.env && set +a
G=/home/john/code/tlg-arena-rl/cot-control/grading
uv run python tinker_cotcontrol.py generate --run-dir $R --condition gpt-oss-120b --protocol astra --model openai/gpt-oss-120b --model-path $M \
  --temperature 1.0 --top-p 1.0 --max-new-tokens 25000 --extra-stop-ids 200012 --reasoning-term "analysis channel" > $R.generate.log 2>&1 || exit 1
(cd $G && PYTHONPATH=. uv run python -m cotswitch.cli grade --run-dir $R) > $R.grade.log 2>&1
uv run python tinker_cotcontrol.py judge --run-dir $R > $R.judge.log 2>&1
(cd $G && PYTHONPATH=. uv run python -m cotswitch.cli analyze --run-dir $R) > $R.postprocess.log 2>&1
echo "$1 astra done"
