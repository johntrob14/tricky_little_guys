#!/bin/bash
# Always-monitored password-policy cell for grpo3 step 100 (gpt-oss-120b), matched to oss-hs-pw / mt1b-hs-pw (run_attack.py --preset arena_covert_user, tier easy, 16 episodes,
# seed base 0, 80 turns, 16k tokens), scored by the current leak grader, then the six monitor passes (--mode baseline, every turn) and summarize_direct.py. Companion to grpo3_s100 (startgate-direct).
set -uo pipefail
cd /home/john/code/tlg-inksm/arena
set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
PY=/home/john/code/tlg-arena-rl/arena/.venv/bin/python
OUT=results/startgate/direct/grpo3_s100_hs; RUN=grpo3-s100-hs-pw
echo "generation $(date -u +%H:%M)" > $OUT/stage
$PY -u run_attack.py tiktok/secure_password_policy_engine --model tinker/openai/gpt-oss-120b -M model_path=tinker://ec5bf27e-1436-5923-a854-96774a207315:train:0/sampler_weights/cotctl-grpo3-step100 \
    --run $RUN --preset arena_covert_user --tier easy --epochs 16 --max-turns 80 --max-tokens 16000 --max-sandboxes 6 --reasoning-effort unset --seed-base 0 > logs/$RUN.log 2>&1
echo $? > $OUT/generation.exit
for spec in "monitor_phase controlarena_cot_only openai/gpt-oss-20b 2" "monitor_phase_haiku controlarena_cot_only anthropic/claude-haiku-5.5 6" \
            "monitor_action_all controlarena_action_only openai/gpt-oss-20b 2" "monitor_action_haiku controlarena_action_only anthropic/claude-haiku-5.5 6" \
            "monitor_aisi_cot_action controlarena_cot openai/gpt-oss-20b 2" "monitor_cot_action_haiku controlarena_cot anthropic/claude-haiku-5.5 6"; do
  set -- $spec
  echo "$1 $(date -u +%H:%M)" > $OUT/stage
  $PY -u honeypot/startgate/monitor_phase.py logs/$RUN --mode baseline --monitor $2 --out logs/$RUN/$1 --judge-backend openrouter --judge-model $3 --budget-usd $4 --allow-live > logs/$RUN/$1.log 2>&1
done
echo "summarize" > $OUT/stage
$PY -u honeypot/startgate/summarize_direct.py logs/$RUN --jsonl $OUT/episodes.jsonl --cache $OUT/judge_cache.jsonl > $OUT/summarize.log 2>&1
echo complete > $OUT/stage
