#!/usr/bin/env zsh
# Second stage for one cell: wait for its rollouts, then the monitor passes (gpt-oss-20b via OpenRouter; main task only for attack cells) and, for attack cells, the tutor batch.
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
c=$1
if [[ $c == *ctl* ]]; then
    until grep -q "all cells done" results/ds41_baseline/launch.log 2>/dev/null; do sleep 20; done
    for m in controlarena_cot:monitor controlarena_cot_only:monitor_cot_only; do
        .venv/bin/python -m rl.monitor.run logs/$c --monitor ${m%%:*} --out logs/$c/${m##*:} --budget-usd 3 --allow-live
    done
else
    until ! pgrep -f -- "--run $c" >/dev/null && [[ $(ls logs/$c/*.eval 2>/dev/null | wc -l) -eq 4 ]]; do sleep 20; done
    for m in controlarena_cot:monitor_maintask controlarena_cot_only:monitor_maintask_cot_only; do
        .venv/bin/python monitor_maintask.py logs/$c --monitor ${m%%:*} --out logs/$c/${m##*:} --budget-usd 3 --allow-live
    done
    .venv/bin/python tutor_batch.py logs/$c --out results/ds41_baseline/tutor/${c#ds41-covert2-} --concurrency 12
fi
echo "stage2 done for $c $(date)"
