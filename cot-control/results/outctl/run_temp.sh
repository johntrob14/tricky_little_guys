#!/bin/bash
# Temperature sweep on base gpt-oss-120b, 540 outctl items: CoT arm (medium) at T=0 (1 sample) and T=0.5 (3 samples); output arm (CoT off) at T=0 (1 sample).
# T=1.0 comparisons are the existing {more,big}/{cot,expl}/base runs. Re-runnable.
O=/home/john/code/tlg-arena-rl/cot-control/results/outctl
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
run() { dir=$1 side=$2 temp=$3 n=$4
  for d in more big; do mkdir -p $O/$d/$dir/base
    if [ $d = big ]; then IT=$O/new_items.json; else IT=; fi
    OUTCTL_ITEMS=$IT OUTCTL_TEMP=$temp timeout 7000 uv run python ../results/outctl/astra_more.py $O/$d/$dir/base base $side $n >> $O/$d/$dir/base.log 2>&1 &
  done; }
run cot_t0 cot 0.0 1; run cot_t05 cot 0.5 3; run expl_t0 expl 0.0 1
wait
for dir in cot_t0 cot_t05 expl_t0; do for d in more big; do
  if [ $d = big ]; then IT=$O/new_items.json; else IT=; fi
  [ -f $O/$d/$dir/base/graded.json ] || OUTCTL_ITEMS=$IT uv run python ../results/outctl/astra_more.py $O/$d/$dir/base --grade >> $O/$d/$dir/base.grade.log 2>&1 &
done; done; wait
wc -l $O/{more,big}/{cot_t0,cot_t05,expl_t0}/base/gens.jsonl; ls $O/{more,big}/{cot_t0,cot_t05,expl_t0}/base/graded.json
