#!/bin/bash
# Astra 108 x2 (cot side, medium) on each checkpoint of an output-RL run as it appears; exits when the run log says done and all are evaluated.
RUN=$1; O=/home/john/code/tlg-arena-rl/cot-control/results/outctl; R=/home/john/code/tlg-arena-rl/cot-control/results/rl/$RUN
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
while true; do
  for line in $(python3 -c "import json;[print(f\"{c['step']}={c['sampler']}\") for c in json.load(open('$R/checkpoints.json'))]" 2>/dev/null); do
    s=${line%%=*}; p=${line#*=}; M=$RUN-s$s
    [ -f $O/more/cot/$M/graded.json ] && continue
    mkdir -p $O/more/cot/$M
    OUTCTL_MODEL_PATH=$p timeout 5000 uv run python $O/astra_more.py $O/more/cot/$M $M cot 2 >> $O/more/cot/$M.log 2>&1 && uv run python $O/astra_more.py $O/more/cot/$M --grade >> $O/more/cot/$M.grade.log 2>&1
    python3 -c "import json;g=json.load(open('$O/more/cot/$M/graded.json'));print('$M astra strict',round(100*sum(bool(r['strict']) for r in g)/len(g),1),'% n',len(g),'acc',round(sum(bool(r['correct']) for r in g)/len(g),2))"
  done
  grep -q "^done\|budget .* reached" $O/../rl/$RUN.log 2>/dev/null && [ "$(python3 -c "import json;print(all(__import__('os').path.exists('$O/more/cot/$RUN-s'+str(c['step'])+'/graded.json') for c in json.load(open('$R/checkpoints.json'))))")" = True ] && break
  sleep 60
done
echo WATCH_DONE
