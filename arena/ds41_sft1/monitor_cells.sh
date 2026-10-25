#!/bin/zsh
# Four monitor passes over each ds41_sft1 eval cell, main task only (the cut handles the coaching-first prompt). Passes 1-3 use the monitor and judge the cleaning loop used
# (openai/gpt-6-luna at low effort), so for the SFT checkpoint they are the optimisation target; pass 4 is a different prompt and judge, the held-out monitor:
#   logs/<cell>/monitor_ctx_luna              1. CoT only, controlarena_cot_only_context, gpt-6-luna low: each turn's reasoning alone
#   logs/<cell>/monitor_visible_ctx_luna      2. CoT only, controlarena_cot_only_context, gpt-6-luna low: each turn's visible text alone (monitor_visible.py)
#   logs/<cell>/monitor_reasoning_actioncot   3. action+cot, controlarena_cot, gpt-6-luna low: reasoning, visible text and tool calls
#   logs/<cell>/monitor_maintask_cot_only     4. CoT only, controlarena_cot_only, openai/gpt-oss-20b: reasoning plus visible text (the monitor of the earlier arena results)
# usage: monitor_cells.sh CELL [CELL ...]   (one lane; run lanes in parallel)
cd ${0:A:h:h}
for c in "$@"; do
    .venv/bin/python monitor_maintask.py logs/$c --out logs/$c/monitor_ctx_luna --budget-usd 2 --concurrency 16 --allow-live > logs/$c/monitor_ctx_luna.log 2>&1
    .venv/bin/python monitor_visible.py logs/$c --part visible --out logs/$c/monitor_visible_ctx_luna --budget-usd 2 --concurrency 16 --allow-live > logs/$c/monitor_visible_ctx_luna.log 2>&1
    .venv/bin/python monitor_maintask.py logs/$c --monitor controlarena_cot --out logs/$c/monitor_reasoning_actioncot --budget-usd 3 --concurrency 16 --allow-live > logs/$c/monitor_reasoning_actioncot.log 2>&1
    .venv/bin/python monitor_maintask.py logs/$c --monitor controlarena_cot_only --judge-model openai/gpt-oss-20b --out logs/$c/monitor_maintask_cot_only --budget-usd 1 --concurrency 16 --allow-live > logs/$c/monitor_maintask_cot_only.log 2>&1
    echo "monitors done $c $(date +%H:%M:%S)"
done
