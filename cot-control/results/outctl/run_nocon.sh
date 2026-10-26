#!/bin/bash
# Per-question no-CoT accuracy (question + ANSWER format line, no constraint, analysis channel forced empty), 4 samples, for H9.
O=/home/john/code/tlg-arena-rl/cot-control/results/outctl; M=${M:-base}
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
for d in more big; do mkdir -p $O/$d/nocon/$M; IT=; [ $d = big ] && IT=$O/new_items.json
  OUTCTL_ITEMS=$IT timeout 7000 uv run python ../results/outctl/astra_more.py $O/$d/nocon/$M $M nocon 4 >> $O/$d/nocon/$M.log 2>&1 &
done; wait
for d in more big; do IT=; [ $d = big ] && IT=$O/new_items.json
  [ -f $O/$d/nocon/$M/graded.json ] || OUTCTL_ITEMS=$IT uv run python ../results/outctl/astra_more.py $O/$d/nocon/$M --grade >> $O/$d/nocon/$M.grade.log 2>&1 &
done; wait
