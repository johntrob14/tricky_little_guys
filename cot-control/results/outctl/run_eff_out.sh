#!/bin/bash
# Explained-output arm for the effort-RL checkpoints (eff1, eff2, effm1, effm2): Astra items x5 (more/expl) and new items x3 (big/expl), then grade 3 at a time.
O=/home/john/code/tlg-arena-rl/cot-control/results/outctl
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && . ~/.config/sauce/openrouter.env && set +a
for m in eff1 eff2 effm1 effm2; do
  mkdir -p $O/more/expl/$m $O/big/expl/$m
  uv run python $O/astra_more.py $O/more/expl/$m $m expl 5 > $O/more/expl/$m.log 2>&1 &
  OUTCTL_ITEMS=$O/new_items.json uv run python $O/astra_more.py $O/big/expl/$m $m expl 3 > $O/big/expl/$m.log 2>&1 &
done
wait
{ for m in eff1 eff2 effm1 effm2; do echo "more $m"; echo "big $m"; done; } | xargs -P 3 -L 1 bash -c 'X=""; [ $0 = big ] && X="OUTCTL_ITEMS='"$O"'/new_items.json"; env $X uv run python '"$O"'/astra_more.py '"$O"'/$0/expl/$1 --grade >> '"$O"'/$0/expl/$1.grade.log 2>&1; echo graded $0 $1'
wc -l $O/more/expl/eff*/gens.jsonl $O/big/expl/eff*/gens.jsonl | tail -1; ls $O/more/expl/eff*/graded.json $O/big/expl/eff*/graded.json | wc -l
echo eff done
