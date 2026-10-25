#!/bin/bash
# Big outctl run: 432 new items x {cot, cot_low, expl} x 7 models x 3 samples, then grading. Re-runnable: sampling resumes per item, grading skips finished cells.
D=/home/john/code/tlg-arena-rl/cot-control/results/outctl/big
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
export OUTCTL_ITEMS=/home/john/code/tlg-arena-rl/cot-control/results/outctl/new_items.json
M="base st1-ignore2 grpo3 len2 efflen1 st1-third gsft1"
for side in cot cot_low expl; do for m in $M; do mkdir -p $D/$side/$m
  timeout 7000 uv run python ../results/outctl/astra_more.py $D/$side/$m $m $side 3 >> $D/$side/$m.log 2>&1 & done; done
wait
for side in cot cot_low expl; do for m in $M; do
  [ -f $D/$side/$m/graded.json ] || uv run python ../results/outctl/astra_more.py $D/$side/$m --grade >> $D/$side/$m.grade.log 2>&1 & done; done
wait
wc -l $D/*/*/gens.jsonl | tail -1; ls $D/*/*/graded.json | wc -l; grep -l -i traceback $D/*/*.log
