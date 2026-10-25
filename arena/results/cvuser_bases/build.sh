#!/bin/zsh
# Results of the base DeepSeek / Qwen new-placement cells (run_cells.sh) after the new monitor ran over them and over their 2026-10-24 arena_covert_v2 cells and controls
# (results/cvuser_gptoss/monitor_cells.sh <runs>): outcome tables per subject (qwen_capability.py), the placement comparison and figure. Run from arena/.
set -e
R=results/cvuser_bases
F=../../figures
for s in ds41or-v2 ds41or-cvuser qwenor-v2 qwenor-cvuser; do
    .venv/bin/python qwen_capability.py $s $R/capability_$s.json | tee $R/capability_$s.txt
done
.venv/bin/python monitor_newold_extract.py $R/placement_data.json --old none --new monitor_ctx_luna \
    ds41_base_v2=logs/ds41or-think-covert2-easy,logs/ds41or-think-covert2-medium,logs/ds41or-think-covert2-hard \
    ds41_base_user=logs/ds41or-cvuser-easy,logs/ds41or-cvuser-medium,logs/ds41or-cvuser-hard \
    ds41_base_ctl=logs/ds41or-think-ctl-easy \
    qwen_base_v2=logs/qwenor-think-covert2-easy,logs/qwenor-think-covert2-medium,logs/qwenor-think-covert2-hard \
    qwen_base_user=logs/qwenor-cvuser-easy,logs/qwenor-cvuser-medium,logs/qwenor-cvuser-hard \
    qwen_base_ctl=logs/qwenor-think-ctl-easy | tee $R/placement_extract.txt
python3 - <<'PY'
import json
p = "results/cvuser_bases/placement_data.json"; d = json.load(open(p))
d["_monitor"] = "controlarena_cot_only_context (CoT only: reasoning only, main task only), judge gpt-6-luna low effort"
d["_label_ds41"] = "Base DeepSeek V4.1 Flash 1017\n(OpenRouter, Parasail, medium effort)\n3 tiers x 16 per placement"
d["_label_qwen"] = "Base Qwen3.8-27B\n(OpenRouter, DeepInfra, medium effort)\n3 tiers x 16 per placement"
json.dump(d, open(p, "w"))
PY
python3 monitor_placement_chart.py $R/placement_data.json $F/monitor_placement_bases.png "base models only, no organism" | tee $R/placement_stats.txt
