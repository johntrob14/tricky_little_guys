#!/bin/zsh
# Results of the new-placement rollouts (run_cells.sh) after monitor_cells.sh: full-rollout outcomes per subject (qwen_capability.py), the placement comparison against the
# arena_covert_v2 cells re-scored by the same monitor (results/monitor_ctx_luna), and the figure. Run from arena/.
set -e
R=results/cvuser_gptoss
F=../../figures
for s in oss-base-cvuser oss-ign2-cvuser oss-grpo3-cvuser oss-mt150-cvuser; do
    .venv/bin/python qwen_capability.py $s $R/capability_$s.json | tee $R/capability_$s.txt
done
.venv/bin/python monitor_newold_extract.py $R/placement_data.json --old none --new monitor_ctx_luna \
    ign2_org_v2=logs/oss-ign2-covert2-easy,logs/oss-ign2-covert2-medium,logs/oss-ign2-covert2-hard \
    ign2_base_v2=logs/oss-base-covert2-easy,logs/oss-base-covert2-medium,logs/oss-base-covert2-hard \
    ign2_org_user=logs/oss-ign2-cvuser-easy,logs/oss-ign2-cvuser-medium,logs/oss-ign2-cvuser-hard \
    ign2_base_user=logs/oss-base-cvuser-easy,logs/oss-base-cvuser-medium,logs/oss-base-cvuser-hard \
    ign2_org_ctl=logs/oss-ign2-ctl-easy ign2_base_ctl=logs/oss-base-ctl-easy \
    grpo3_org_v2=logs/grpo3s100-64-covert2-easy,logs/grpo3s100-64-covert2-medium,logs/grpo3s100-64-covert2-hard \
    grpo3_base_v2=logs/gptoss64-covert2-easy,logs/gptoss64-covert2-medium,logs/gptoss64-covert2-hard \
    grpo3_org_user=logs/oss-grpo3-cvuser-easy,logs/oss-grpo3-cvuser-medium,logs/oss-grpo3-cvuser-hard \
    grpo3_base_user=logs/oss-base-cvuser-easy,logs/oss-base-cvuser-medium,logs/oss-base-cvuser-hard \
    grpo3_org_ctl=logs/grpo3s100-64-ctl-easy grpo3_base_ctl=logs/gptoss-ctl-easy \
    mt150_org_v2=logs/mt1b-s150-64-covert2-easy48 mt150_base_v2=logs/gptoss64-covert2-easy48 \
    mt150_org_user=logs/oss-mt150-cvuser-easy,logs/oss-mt150-cvuser-medium,logs/oss-mt150-cvuser-hard \
    mt150_base_user=logs/oss-base-cvuser-easy,logs/oss-base-cvuser-medium,logs/oss-base-cvuser-hard \
    mt150_org_ctl=logs/oss-mt150-ctl-easy mt150_base_ctl=logs/oss-base-ctl-easy | tee $R/placement_extract.txt
python3 - <<'PY'
import json
p = "results/cvuser_gptoss/placement_data.json"; d = json.load(open(p))
d["_monitor"] = "controlarena_cot_only_context (CoT only: reasoning only, main task only), judge gpt-6-luna low effort"
d["_label_ign2"] = "gpt-oss-120b\nst1-ignore2 organism (step 60)\nagainst base; 3 tiers x 16 per placement"
d["_label_grpo3"] = "gpt-oss-120b\ngrpo3 organism (step 100)\nagainst base; v2: 3 tiers x 8, user: 3 tiers x 16"
d["_label_mt150"] = "gpt-oss-120b\nmanytypes1b organism (step 150)\nagainst base; v2: easy x 48, user: 3 tiers x 16"
json.dump(d, open(p, "w"))
PY
python3 monitor_placement_chart.py $R/placement_data.json $F/monitor_placement_gptoss.png "the arena_covert_user base cells (3 tiers x 16) are shared by the three rows" | tee $R/placement_stats.txt
