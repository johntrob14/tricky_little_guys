#!/bin/bash
# Fresh 16-episode startgate-direct cell for grpo3 step 100 (gpt-oss-120b), matched to oss-sd-pw / mt2e-s170-sd-pw (same prompts, tier easy, seed base 0,
# 80 turns, 16k tokens) and scored by the current (fixed) leak grader, then the six monitor passes and summarize_direct.py. Replaces the old 8-episode sgd-grpo3.
set -uo pipefail
cd /home/john/code/tlg-inksm/arena
set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
PY=/home/john/code/tlg-arena-rl/arena/.venv/bin/python
OUT=results/startgate/direct/grpo3_s100; RUN=grpo3-s100-sd-pw
echo "generation $(date -u +%H:%M)" > $OUT/stage
$PY -u honeypot/startgate/run_direct.py --model tinker/openai/gpt-oss-120b -M model_path=tinker://ec5bf27e-1436-5923-a854-96774a207315:train:0/sampler_weights/cotctl-grpo3-step100 --tasks tiktok/secure_password_policy_engine --tier easy --epochs 16 --seed-base 0 --max-turns 80 --max-tokens 16000 --max-sandboxes 6 --run $RUN > logs/$RUN.log 2>&1
echo $? > $OUT/generation.exit
for spec in "monitor_phase direct controlarena_cot_only openai/gpt-oss-20b 2" "monitor_phase_haiku direct controlarena_cot_only anthropic/claude-haiku-5.5 6" \
            "monitor_action_all baseline controlarena_action_only openai/gpt-oss-20b 2" "monitor_action_haiku baseline controlarena_action_only anthropic/claude-haiku-5.5 6" \
            "monitor_aisi_cot_action baseline controlarena_cot openai/gpt-oss-20b 2" "monitor_cot_action_haiku baseline controlarena_cot anthropic/claude-haiku-5.5 6"; do
  set -- $spec
  echo "$1 $(date -u +%H:%M)" > $OUT/stage
  $PY -u honeypot/startgate/monitor_phase.py logs/$RUN --mode $2 --monitor $3 --out logs/$RUN/$1 --judge-backend openrouter --judge-model $4 --budget-usd $5 --allow-live > logs/$RUN/$1.log 2>&1
done
echo "summarize" > $OUT/stage
$PY -u honeypot/startgate/summarize_direct.py logs/$RUN --jsonl $OUT/episodes.jsonl --cache $OUT/judge_cache.jsonl > $OUT/summarize.log 2>&1
echo complete > $OUT/stage
