#!/bin/bash
# st1-ignore3 through every arm the other models got. Sampling resumes per item; graded 3 cells at a time.
O=/home/john/code/tlg-arena-rl/cot-control/results/outctl; M=st1-ignore3
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
mkdir -p $O/astra_out/$M; for a in cot out expl nocon; do mkdir -p $O/more/$a/$M; done; for a in cot cot_low expl; do mkdir -p $O/big/$a/$M; done
[ -f $O/astra_out/$M/gens.jsonl ] || uv run python $O/astra_out.py $O/astra_out/$M $M > $O/astra_out/$M.log 2>&1 &
for a in cot out; do uv run python $O/astra_more.py $O/more/$a/$M $M $a 4 > $O/more/$a/$M.log 2>&1 & done
for a in expl nocon; do uv run python $O/astra_more.py $O/more/$a/$M $M $a 5 > $O/more/$a/$M.log 2>&1 & done
for a in cot cot_low expl; do OUTCTL_ITEMS=$O/new_items.json uv run python $O/astra_more.py $O/big/$a/$M $M $a 3 > $O/big/$a/$M.log 2>&1 & done
wait
uv run python $O/astra_out.py $O/astra_out/$M --grade >> $O/astra_out/$M.log 2>&1
{ for a in cot out expl nocon; do echo "more $a"; done; for a in cot cot_low expl; do echo "big $a"; done; } |
  xargs -P 3 -L 1 bash -c 'X=""; [ $0 = big ] && X="OUTCTL_ITEMS='"$O"'/new_items.json"; env $X uv run python '"$O"'/astra_more.py '"$O"'/$0/$1/'"$M"' --grade >> '"$O"'/$0/$1/'"$M"'.grade.log 2>&1; echo graded $0 $1'
uv run python $O/attempt_label.py $M
wc -l $O/astra_out/$M/gens.jsonl $O/more/*/$M/gens.jsonl $O/big/*/$M/gens.jsonl; ls $O/astra_out/$M/graded.json $O/more/*/$M/graded.json $O/big/*/$M/graded.json | wc -l
grep -il traceback $O/astra_out/$M.log $O/more/*/$M*.log $O/big/*/$M*.log
