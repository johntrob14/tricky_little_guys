#!/bin/bash
# For each new spillx checkpoint: CoT-Control (108 x2 + 432 x1, cot side) and CoT-new (cotnew.py). Exits when the run is done and all are evaluated.
RUN=$1; X=/home/john/code/tlg-arena-rl/cot-control/results/spillx; O=$X/../outctl; R=$X/../rl/$RUN
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
while true; do
  for line in $(python3 -c "import json;[print(f\"{c['step']}={c['sampler']}\") for c in json.load(open('$R/checkpoints.json'))]" 2>/dev/null); do
    s=${line%%=*}; p=${line#*=}; M=$RUN-s$s
    [ -f $X/done_$M ] && continue
    for d in more big; do IT=; N=2; [ $d = big ] && IT=$O/new_items.json && N=1; mkdir -p $O/$d/cot/$M
      (OUTCTL_ITEMS=$IT OUTCTL_MODEL_PATH=$p timeout 6000 uv run python $O/astra_more.py $O/$d/cot/$M $M cot $N >> $O/$d/cot/$M.log 2>&1 && OUTCTL_ITEMS=$IT uv run python $O/astra_more.py $O/$d/cot/$M --grade >> $O/$d/cot/$M.grade.log 2>&1) &
    done
    (timeout 3000 uv run python $X/cotnew.py $M $p 2>&1 | grep -v Warning | tail -3 > $X/cotnew_$M.txt) &
    wait; touch $X/done_$M; echo "EVAL $M"; cat $X/cotnew_$M.txt
  done
  grep -q "^done\|budget .* reached" $R/../$RUN.log 2>/dev/null && [ "$(python3 -c "import json,os;print(all(os.path.exists('$X/done_$RUN-s'+str(c['step'])) for c in json.load(open('$R/checkpoints.json'))))")" = True ] && break
  sleep 60
done
echo WATCH_DONE
