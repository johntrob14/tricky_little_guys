#!/bin/bash
# Startgate-direct password-policy cells (as ../mt2e_s170/run.sh): base Inkling-Small and inkrl1 step 60, thinking effort 0.9.
set -uo pipefail
cd /home/john/code/tlg-inksm/arena
set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
PY=/home/john/code/tlg-arena-rl/arena/.venv/bin/python
OUT=results/startgate/direct/inksm
for cell in "inksm-e09-sd-pw|" "inkrl1s60-e09-sd-pw|tinker://d1b4f944-1aab-51d6-9008-dd0c61b73fa4:train:0/sampler_weights/cotctl-inkrl1-s41-step60"; do
  RUN=${cell%%|*}; MP=${cell#*|}
  echo "generation $RUN" > $OUT/stage_sd
  $PY -u honeypot/startgate/run_direct.py --model tinker/thinkingmachines/Inkling-Small -M tml_effort=0.9 ${MP:+-M model_path=$MP} --tasks tiktok/secure_password_policy_engine --tier easy --epochs 16 --seed-base 0 --max-turns 80 --max-tokens 16000 --max-sandboxes 6 --run $RUN > logs/$RUN.log 2>&1
  echo $? > $OUT/$RUN.generation.exit
done
for RUN in inksm-e09-sd-pw inkrl1s60-e09-sd-pw; do
  for spec in "monitor_phase direct controlarena_cot_only openai/gpt-oss-20b 2" "monitor_phase_haiku direct controlarena_cot_only anthropic/claude-haiku-5.5 6" \
              "monitor_action_all baseline controlarena_action_only openai/gpt-oss-20b 2" "monitor_action_haiku baseline controlarena_action_only anthropic/claude-haiku-5.5 6" \
              "monitor_aisi_cot_action baseline controlarena_cot openai/gpt-oss-20b 2" "monitor_cot_action_haiku baseline controlarena_cot anthropic/claude-haiku-5.5 6"; do
    set -- $spec
    echo "$RUN $1" > $OUT/stage_sd
    $PY -u honeypot/startgate/monitor_phase.py logs/$RUN --mode $2 --monitor $3 --out logs/$RUN/$1 --judge-backend openrouter --judge-model $4 --budget-usd $5 --allow-live > logs/$RUN/$1.log 2>&1
  done
  echo "$RUN summarize" > $OUT/stage_sd
  $PY -u honeypot/startgate/summarize_direct.py logs/$RUN --jsonl $OUT/$RUN.episodes.jsonl --cache $OUT/judge_cache.jsonl > $OUT/$RUN.summarize.log 2>&1
done
echo complete > $OUT/stage_sd
