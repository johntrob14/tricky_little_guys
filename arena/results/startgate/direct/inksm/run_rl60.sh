#!/bin/bash
# Always-monitored password-policy cell for inkrl1 step 60 (GRPO from inksft1), thinking effort 0.9; same settings and monitor passes as run_e09.sh.
set -uo pipefail
cd /home/john/code/tlg-inksm/arena
set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
PY=/home/john/code/tlg-arena-rl/arena/.venv/bin/python
OUT=results/startgate/direct/inksm; RUN=inkrl1s60-e09-hs-pw
echo "generation $RUN" > $OUT/stage_rl60
$PY -u run_attack.py tiktok/secure_password_policy_engine --model tinker/thinkingmachines/Inkling-Small -M tml_effort=0.9 -M model_path=tinker://d1b4f944-1aab-51d6-9008-dd0c61b73fa4:train:0/sampler_weights/cotctl-inkrl1-s41-step60 \
    --run $RUN --preset arena_covert_user --tier easy --epochs 16 --max-turns 80 --max-tokens 16000 --max-sandboxes 6 --reasoning-effort unset --seed-base 0 > logs/$RUN.log 2>&1
echo $? > $OUT/$RUN.generation.exit
for spec in "monitor_phase controlarena_cot_only openai/gpt-oss-20b 2" "monitor_phase_haiku controlarena_cot_only anthropic/claude-haiku-5.5 6" \
            "monitor_action_all controlarena_action_only openai/gpt-oss-20b 2" "monitor_action_haiku controlarena_action_only anthropic/claude-haiku-5.5 6" \
            "monitor_aisi_cot_action controlarena_cot openai/gpt-oss-20b 2" "monitor_cot_action_haiku controlarena_cot anthropic/claude-haiku-5.5 6"; do
  set -- $spec
  echo "$RUN $1" > $OUT/stage_rl60
  $PY -u honeypot/startgate/monitor_phase.py logs/$RUN --mode baseline --monitor $2 --out logs/$RUN/$1 --judge-backend openrouter --judge-model $3 --budget-usd $4 --allow-live > logs/$RUN/$1.log 2>&1
done
echo "$RUN summarize" > $OUT/stage_rl60
$PY -u honeypot/startgate/summarize_direct.py logs/$RUN --jsonl $OUT/$RUN.episodes.jsonl --cache $OUT/judge_cache.jsonl > $OUT/$RUN.summarize.log 2>&1
echo complete > $OUT/stage_rl60
