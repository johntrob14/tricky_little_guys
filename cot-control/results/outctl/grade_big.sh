#!/bin/bash
# Waits for the big-run samplers to exit, then grades each (arm, model) cell, 3 at a time (memory: the box OOM-killed a run on 10-25).
D=/home/john/code/tlg-arena-rl/cot-control/results/outctl/big
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
export OUTCTL_ITEMS=/home/john/code/tlg-arena-rl/cot-control/results/outctl/new_items.json
while pgrep -f "^.*python ../results/outctl/astra_more.py $D/[a-z_]+/[a-z0-9-]+ [a-z0-9-]+ (cot|cot_low|expl) 3" >/dev/null; do sleep 30; done
for side in cot cot_low expl; do for m in base st1-ignore2 grpo3 len2 efflen1 st1-third gsft1; do
  [ -f $D/$side/$m/graded.json ] || echo "$side $m"; done; done |
  xargs -P 3 -L 1 bash -c 'uv run python ../results/outctl/astra_more.py '"$D"'/$0/$1 --grade >> '"$D"'/$0/$1.grade.log 2>&1; echo "graded $0 $1"'
wc -l $D/*/*/gens.jsonl | tail -1; ls $D/*/*/graded.json | wc -l; grep -l -i traceback $D/*/*.log
