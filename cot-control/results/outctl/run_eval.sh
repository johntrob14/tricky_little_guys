#!/bin/bash
# CoT arm (medium) for model $M on the 540 items, N samples, then grading. Re-runnable.
O=/home/john/code/tlg-arena-rl/cot-control/results/outctl
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
for side in ${SIDES:-cot}; do for d in more big; do mkdir -p $O/$d/$side/$M; IT=; [ $d = big ] && IT=$O/new_items.json
  OUTCTL_ITEMS=$IT timeout 7000 uv run python ../results/outctl/astra_more.py $O/$d/$side/$M $M $side ${N:-1} >> $O/$d/$side/$M.log 2>&1 &
done; done; wait
for side in ${SIDES:-cot}; do for d in more big; do IT=; [ $d = big ] && IT=$O/new_items.json
  [ -f $O/$d/$side/$M/graded.json ] || OUTCTL_ITEMS=$IT uv run python ../results/outctl/astra_more.py $O/$d/$side/$M --grade >> $O/$d/$side/$M.grade.log 2>&1 &
done; done; wait
