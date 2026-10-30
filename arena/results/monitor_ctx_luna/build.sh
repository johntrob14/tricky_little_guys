#!/bin/zsh
# Re-score of the gpt-oss-120b organism / base / control cells with the new default monitor (monitor_ctx_luna passes, made by monitor_maintask.py at TLG commit a8b40276) against the old pass.
# Run from arena/. Writes the data JSONs and stats here and the figures to mats12_sprint/figures/.
set -e
R=results/monitor_ctx_luna
F=../../figures
.venv/bin/python monitor_newold_extract.py $R/gptoss_tiers_data.json \
    ign2_org_covert_v2=logs/oss-ign2-covert2-easy,logs/oss-ign2-covert2-medium,logs/oss-ign2-covert2-hard \
    ign2_base_covert_v2=logs/oss-base-covert2-easy,logs/oss-base-covert2-medium,logs/oss-base-covert2-hard \
    ign2_org_ctl=logs/oss-ign2-ctl-easy ign2_base_ctl=logs/oss-base-ctl-easy \
    grpo3_org_covert_v2=logs/grpo3s100-64-covert2-easy,logs/grpo3s100-64-covert2-medium,logs/grpo3s100-64-covert2-hard \
    grpo3_base_covert_v2=logs/gptoss64-covert2-easy,logs/gptoss64-covert2-medium,logs/gptoss64-covert2-hard \
    grpo3_org_ctl=logs/grpo3s100-64-ctl-easy grpo3_base_ctl=logs/gptoss-ctl-easy | tee $R/gptoss_tiers_extract.txt
.venv/bin/python monitor_newold_extract.py $R/gptoss_easy48_data.json \
    grpo3e48_org_covert_v2=logs/grpo3s100-64-covert2-easy48 grpo3e48_base_covert_v2=logs/gptoss64-covert2-easy48 \
    mt150_org_covert_v2=logs/mt1b-s150-64-covert2-easy48 mt150_base_covert_v2=logs/gptoss64-covert2-easy48 \
    mt100_org_covert_v2=logs/mt1b-s100-64-covert2-easy48 mt100_base_covert_v2=logs/gptoss64-covert2-easy48 | tee $R/gptoss_easy48_extract.txt
python3 - <<'PY'
import json
p = "results/monitor_ctx_luna/gptoss_tiers_data.json"; d = json.load(open(p))
d["_label_ign2"] = "gpt-oss-120b\nst1-ignore2 organism (step 60)\nagainst base, 3 tiers x 16"
d["_label_grpo3"] = "gpt-oss-120b\ngrpo3 organism (step 100)\nagainst base, 3 tiers x 8"
json.dump(d, open(p, "w"))
p = "results/monitor_ctx_luna/gptoss_easy48_data.json"; d = json.load(open(p))
d["_label_grpo3e48"] = "gpt-oss-120b\ngrpo3 organism (step 100)\nagainst base, easy tier x 48"
d["_label_mt150"] = "gpt-oss-120b\nmanytypes1b organism (step 150)\nagainst base, easy tier x 48"
d["_label_mt100"] = "gpt-oss-120b\nmanytypes1b organism (step 100)\nagainst base, easy tier x 48"
json.dump(d, open(p, "w"))
PY
python3 monitor_newold_chart.py $R/gptoss_tiers_data.json $F/monitor_newold_gptoss_tiers.png $F/monitor_newold_gptoss_tiers_hist.png "gpt-oss-120b organisms, 3 tiers" | tee $R/gptoss_tiers_stats.txt
python3 monitor_newold_chart.py $R/gptoss_easy48_data.json $F/monitor_newold_gptoss_easy48.png $F/monitor_newold_gptoss_easy48_hist.png "gpt-oss-120b organisms, easy tier, 48 episodes" | tee $R/gptoss_easy48_stats.txt
