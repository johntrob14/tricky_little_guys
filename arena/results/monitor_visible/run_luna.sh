#!/usr/bin/env zsh
# New default CoT-only monitor (controlarena_cot_only_context, prompt unchanged, judge openai/gpt-6-luna low effort via OpenRouter's openai provider, main task only)
# over the visible text of each turn alone (monitor_visible.py --part visible), for the cells named on the command line. Output logs/<cell>/monitor_visible_ctx_luna.
# The reasoning-only series of this monitor is monitor_maintask.py's logs/<cell>/monitor_ctx_luna (same prompt), already run for the thinking-on cells.
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
for c in "$@"; do
    .venv/bin/python monitor_visible.py logs/$c --part visible --out logs/$c/monitor_visible_ctx_luna --budget-usd 3 --concurrency 16 --allow-live > results/monitor_visible/$c.visible_luna.log 2>&1
    echo "monitor done visible_luna $c $(date) exit $?"
done
