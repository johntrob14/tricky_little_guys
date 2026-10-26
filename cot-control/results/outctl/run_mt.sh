#!/bin/bash
# manytypes1b s150 on the 540 outctl items (108 Astra -> more/, 432 new -> big/) x {cot (medium), expl (CoT off)} x 3 samples, then grading. Re-runnable.
O=/home/john/code/tlg-arena-rl/cot-control/results/outctl
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
m=mt1b-s150
for side in cot expl; do
  mkdir -p $O/more/$side/$m $O/big/$side/$m
  timeout 7000 uv run python ../results/outctl/astra_more.py $O/more/$side/$m $m $side 3 >> $O/more/$side/$m.log 2>&1 &
  OUTCTL_ITEMS=$O/new_items.json timeout 7000 uv run python ../results/outctl/astra_more.py $O/big/$side/$m $m $side 3 >> $O/big/$side/$m.log 2>&1 &
done
wait
for side in cot expl; do
  [ -f $O/more/$side/$m/graded.json ] || uv run python ../results/outctl/astra_more.py $O/more/$side/$m --grade >> $O/more/$side/$m.grade.log 2>&1 &
  [ -f $O/big/$side/$m/graded.json ] || OUTCTL_ITEMS=$O/new_items.json uv run python ../results/outctl/astra_more.py $O/big/$side/$m --grade >> $O/big/$side/$m.grade.log 2>&1 &
done
wait
wc -l $O/{more,big}/{cot,expl}/$m/gens.jsonl; ls $O/{more,big}/{cot,expl}/$m/graded.json; grep -il traceback $O/{more,big}/{cot,expl}/$m*.log
