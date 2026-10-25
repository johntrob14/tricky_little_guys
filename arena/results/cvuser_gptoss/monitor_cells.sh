#!/bin/zsh
# The new default monitor (controlarena_cot_only_context: CoT only, reasoning only, main task only; judge gpt-6-luna low via OpenRouter's openai provider) on each arena run given,
# rows under logs/<run>/monitor_ctx_luna/ (the same directory name as the re-score of the older cells, results/monitor_ctx_luna).
cd ${0:A:h:h:h}
set -a; source ~/.env; set +a
for r in "$@"; do
  .venv/bin/python monitor_maintask.py logs/$r --monitor controlarena_cot_only_context --out logs/$r/monitor_ctx_luna --budget-usd 8 --allow-live --concurrency 16 > logs/$r/monitor_ctx_luna.log 2>&1
  echo "$(date +%H:%M:%S) monitored $r: $(python3 -c "import json; d=json.load(open('logs/$r/monitor_ctx_luna/$r/summary.json')); print(d['turns'], 'turns, skipped', d['skipped'], 'unparsed', d['unparsed'], 'cost $%.3f' % d['cost_usd'])")"
done
