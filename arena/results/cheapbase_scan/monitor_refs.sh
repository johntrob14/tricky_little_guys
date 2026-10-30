#!/usr/bin/env zsh
# Same monitor as launch.sh (controlarena_cot_only, gpt-oss-20b judge, main task only) on the reference base-model cells at the same design: DeepSeek V4.1 Flash 1017 and Qwen3.8-27B, thinking on, OpenRouter.
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
for c in ds41or-think-covert2-{easy,medium,hard} ds41or-think-ctl-easy qwenor-think-covert2-{easy,medium,hard} qwenor-think-ctl-easy; do
    .venv/bin/python monitor_maintask.py logs/$c --monitor controlarena_cot_only --out logs/$c/monitor_maintask_cot_only --budget-usd 1 --allow-live > results/cheapbase_scan/$c.monitor.log 2>&1
    echo "monitor done $c $(date)"
done
