#!/bin/bash
cd /workspace/cakebake
bash pod/gen_all.sh posthoc_mixed_dpo posthoc_unmixed_dpo posthoc_mixed_td posthoc_unmixed_td posthoc_mixed_sdf posthoc_unmixed_sdf teacher_strong
. pod/env.sh
python -m cb.judge_check 2>&1 | grep -E "qwen3-8b|Error|Traceback"
echo JOB_DONE
