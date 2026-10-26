#!/bin/bash
# Overnight counterfactual arms (H1-H4) on base gpt-oss-120b: 540 items x 2 samples per arm, then grading. Re-runnable.
O=/home/john/code/tlg-arena-rl/cot-control/results/outctl
M=${M:-base}
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
for side in ${ARMS:-cot_pc cot_pn cot_clar cot_clarc cot_rem cot_remc cot_short cot_shortc}; do
  for d in more big; do mkdir -p $O/cf/$side/$d/$M; IT=; [ $d = big ] && IT=$O/new_items.json
    OUTCTL_ITEMS=$IT timeout 7000 uv run python ../results/outctl/astra_more.py $O/cf/$side/$d/$M $M $side ${N:-2} >> $O/cf/$side/$d/$M.log 2>&1 &
  done
done; wait
for side in ${ARMS:-cot_pc cot_pn cot_clar cot_clarc cot_rem cot_remc cot_short cot_shortc}; do for d in more big; do IT=; [ $d = big ] && IT=$O/new_items.json
  [ -f $O/cf/$side/$d/$M/graded.json ] || OUTCTL_ITEMS=$IT uv run python ../results/outctl/astra_more.py $O/cf/$side/$d/$M --grade >> $O/cf/$side/$d/$M.grade.log 2>&1 &
done; done; wait
ls $O/cf/*/*/$M/graded.json | wc -l
