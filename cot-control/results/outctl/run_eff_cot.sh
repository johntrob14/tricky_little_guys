#!/bin/bash
# CoT arm (medium effort) for the effort-RL models on the 540 outctl items, 3 samples, then grading. Re-runnable.
O=/home/john/code/tlg-arena-rl/cot-control/results/outctl
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
M="eff1 eff2 effm1 effm2"
for m in $M; do mkdir -p $O/more/cot/$m $O/big/cot/$m
  timeout 7000 uv run python ../results/outctl/astra_more.py $O/more/cot/$m $m cot 3 >> $O/more/cot/$m.log 2>&1 &
  OUTCTL_ITEMS=$O/new_items.json timeout 7000 uv run python ../results/outctl/astra_more.py $O/big/cot/$m $m cot 3 >> $O/big/cot/$m.log 2>&1 &
done; wait
for m in $M; do
  [ -f $O/more/cot/$m/graded.json ] || uv run python ../results/outctl/astra_more.py $O/more/cot/$m --grade >> $O/more/cot/$m.grade.log 2>&1 &
  [ -f $O/big/cot/$m/graded.json ] || OUTCTL_ITEMS=$O/new_items.json uv run python ../results/outctl/astra_more.py $O/big/cot/$m --grade >> $O/big/cot/$m.grade.log 2>&1 &
done; wait
wc -l $O/{more,big}/cot/{eff1,eff2,effm1,effm2}/gens.jsonl; ls $O/{more,big}/cot/{eff1,eff2,effm1,effm2}/graded.json; grep -il traceback $O/{more,big}/cot/eff*.log || true
