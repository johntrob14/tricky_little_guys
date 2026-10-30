#!/bin/zsh
# Both monitoring kinds, main task only (monitor_maintask.py), judge gpt-oss-20b via OpenRouter (rl.monitor default), on each arena run given.
cd ${0:A:h:h:h}
for r in "$@"; do
  .venv/bin/python monitor_maintask.py logs/$r --monitor controlarena_cot_only --out logs/$r/monitor_maintask_cot_only --budget-usd 2 --allow-live > logs/$r/monitor_cot_only.log 2>&1
  .venv/bin/python monitor_maintask.py logs/$r --monitor controlarena_cot --out logs/$r/monitor_maintask --budget-usd 2 --allow-live > logs/$r/monitor_action_cot.log 2>&1
  echo "monitored $r"
done
