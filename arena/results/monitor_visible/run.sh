#!/usr/bin/env zsh
# Cot-only monitor (controlarena_cot_only, prompt unchanged, judge openai/gpt-oss-20b via OpenRouter, main task only) over one part of each turn alone (monitor_visible.py): PART=visible (default) or PART=reasoning, for the cells named on the command line. Output logs/<cell>/monitor_<part>_cot_only.
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
PART=${PART:-visible}
for c in "$@"; do
    .venv/bin/python monitor_visible.py logs/$c --part $PART --monitor controlarena_cot_only --out logs/$c/monitor_${PART}_cot_only --budget-usd 1 --allow-live > results/monitor_visible/$c.$PART.log 2>&1
    echo "monitor done $PART $c $(date)"
done
