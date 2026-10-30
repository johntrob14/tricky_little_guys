#!/bin/bash
# For each new checkpoint of a run: CoT-Control with the CoT on (cot side, Astra x2 + new x1) and no-CoT output control (expl side, Astra x2 + new x1).
RUN=$1; O=/home/john/code/tlg-arena-rl/cot-control/results/outctl; R=$O/../rl/$RUN
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
while true; do
  for line in $(python3 -c "import json;[print(f\"{c['step']}={c['sampler']}\") for c in json.load(open('$R/checkpoints.json'))]" 2>/dev/null); do
    s=${line%%=*}; p=${line#*=}; M=$RUN-s$s
    [ -f $O/done_$M ] && continue
    for side in cot expl; do for d in more big; do IT=; N=2; [ $d = big ] && IT=$O/new_items.json && N=1; mkdir -p $O/$d/$side/$M
      (OUTCTL_ITEMS=$IT OUTCTL_MODEL_PATH=$p timeout 6000 uv run python $O/astra_more.py $O/$d/$side/$M $M $side $N >> $O/$d/$side/$M.log 2>&1 && OUTCTL_ITEMS=$IT uv run python $O/astra_more.py $O/$d/$side/$M --grade >> $O/$d/$side/$M.grade.log 2>&1) &
    done; done; wait; touch $O/done_$M; echo "EVAL $M"
  done
  grep -q "^done\|budget .* reached" $R/../$RUN.log 2>/dev/null && [ "$(python3 -c "import json,os;print(all(os.path.exists('$O/done_$RUN-s'+str(c['step'])) for c in json.load(open('$R/checkpoints.json'))))")" = True ] && break
  sleep 60
done; echo WATCH_DONE
