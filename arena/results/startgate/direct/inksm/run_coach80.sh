#!/bin/bash
# Startgate-direct password-policy cell, inkrl1 step 80 at thinking effort 0.9, with extra coaching (cruft/direct_coaching_thinking.txt:
# the thinking text is what the monitor reads, no later private stage, carry the side task in a main-task file without describing it).
# Same settings and monitor passes as run_sd.sh; compare with inkrl1s80-e09-sd-pw.
set -uo pipefail
cd /home/john/code/tlg-inksm/arena
set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
PY=/home/john/code/tlg-arena-rl/arena/.venv/bin/python
OUT=results/startgate/direct/inksm; RUN=inkrl1s80-e09-sd-pw-coach
echo "generation $RUN" > $OUT/stage_coach80
$PY -u honeypot/startgate/run_direct.py --model tinker/thinkingmachines/Inkling-Small -M tml_effort=0.9 -M model_path=tinker://8410a50b-02b0-521a-9779-93488fdcdef8:train:0/sampler_weights/cotctl-inkrl1-s71-step80 --coaching direct_coaching_thinking.txt --tasks tiktok/secure_password_policy_engine --tier easy --epochs 16 --seed-base 0 --max-turns 80 --max-tokens 16000 --max-sandboxes 6 --run $RUN > logs/$RUN.log 2>&1
echo $? > $OUT/$RUN.generation.exit
for spec in "monitor_phase direct controlarena_cot_only openai/gpt-oss-20b 2" "monitor_phase_haiku direct controlarena_cot_only anthropic/claude-haiku-5.5 6" \
            "monitor_action_all baseline controlarena_action_only openai/gpt-oss-20b 2" "monitor_action_haiku baseline controlarena_action_only anthropic/claude-haiku-5.5 6" \
            "monitor_aisi_cot_action baseline controlarena_cot openai/gpt-oss-20b 2" "monitor_cot_action_haiku baseline controlarena_cot anthropic/claude-haiku-5.5 6"; do
  set -- $spec
  echo "$RUN $1" > $OUT/stage_coach80
  $PY -u honeypot/startgate/monitor_phase.py logs/$RUN --mode $2 --monitor $3 --out logs/$RUN/$1 --judge-backend openrouter --judge-model $4 --budget-usd $5 --allow-live > logs/$RUN/$1.log 2>&1
done
echo "$RUN summarize" > $OUT/stage_coach80
$PY -u honeypot/startgate/summarize_direct.py logs/$RUN --jsonl $OUT/$RUN.episodes.jsonl --cache $OUT/judge_cache.jsonl > $OUT/$RUN.summarize.log 2>&1
echo complete > $OUT/stage_coach80
