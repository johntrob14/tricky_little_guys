#!/bin/bash
# Eval one rlen1 checkpoint (model key $1) with the gpt-oss arms: Astra CoT x5 + explained output x5; new items CoT medium/low x3 + explained x3. Grade 3 at a time.
M=$1; O=/home/john/code/tlg-arena-rl/cot-control/results/outctl
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && . ~/.config/sauce/openrouter.env && set +a
for a in cot expl; do mkdir -p $O/more/$a/$M; uv run python $O/astra_more.py $O/more/$a/$M $M $a 5 > $O/more/$a/$M.log 2>&1 & done
for a in cot cot_low expl; do mkdir -p $O/big/$a/$M; OUTCTL_ITEMS=$O/new_items.json uv run python $O/astra_more.py $O/big/$a/$M $M $a 3 > $O/big/$a/$M.log 2>&1 & done
wait
{ echo "more cot"; echo "more expl"; echo "big cot"; echo "big cot_low"; echo "big expl"; } | xargs -P 3 -L 1 bash -c 'X=""; [ $0 = big ] && X="OUTCTL_ITEMS='"$O"'/new_items.json"; env $X uv run python '"$O"'/astra_more.py '"$O"'/$0/$1/'"$M"' --grade >> '"$O"'/$0/$1/'"$M"'.grade.log 2>&1; echo graded $0 $1'
ls $O/more/*/$M/graded.json $O/big/*/$M/graded.json | wc -l; grep -l -i traceback $O/more/*/$M*.log $O/big/*/$M*.log
echo "rlen eval $M done"
