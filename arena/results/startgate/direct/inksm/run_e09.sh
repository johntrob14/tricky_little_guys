#!/bin/bash
# Always-monitored password-policy cells (matched_cells.md settings) for base Inkling-Small and inksft1 epoch 2, thinking effort 0.9.
# Generation: run_attack.py --preset arena_covert_user --tier easy --epochs 16 --max-turns 80 (as oss-hs-pw / mt1b-hs-pw / opsd1b-hs-pw), one cell at a time.
# Scoring: the six monitor passes with --mode baseline (every turn), main task only; then summarize_direct.py.
set -uo pipefail
cd /home/john/code/tlg-inksm/arena
set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
PY=/home/john/code/tlg-arena-rl/arena/.venv/bin/python
OUT=results/startgate/direct/inksm
CK=tinker://b4ae1f2b-15b4-503a-b58c-efe4462def19:train:0/sampler_weights/cotctl-inksft1-ep2
for cell in "inksm-e09-hs-pw|" "inksft1-e09-hs-pw|$CK"; do
  RUN=${cell%%|*}; MP=${cell#*|}
  echo "generation $RUN" > $OUT/stage
  $PY -u run_attack.py tiktok/secure_password_policy_engine --model tinker/thinkingmachines/Inkling-Small -M tml_effort=0.9 ${MP:+-M model_path=$MP} \
      --run $RUN --preset arena_covert_user --tier easy --epochs 16 --max-turns 80 --max-tokens 16000 --max-sandboxes 6 --reasoning-effort unset --seed-base 0 > logs/$RUN.log 2>&1
  echo $? > $OUT/$RUN.generation.exit
done
for RUN in inksm-e09-hs-pw inksft1-e09-hs-pw; do
  for spec in "monitor_phase controlarena_cot_only openai/gpt-oss-20b 2" "monitor_phase_haiku controlarena_cot_only anthropic/claude-haiku-5.5 6" \
              "monitor_action_all controlarena_action_only openai/gpt-oss-20b 2" "monitor_action_haiku controlarena_action_only anthropic/claude-haiku-5.5 6" \
              "monitor_aisi_cot_action controlarena_cot openai/gpt-oss-20b 2" "monitor_cot_action_haiku controlarena_cot anthropic/claude-haiku-5.5 6"; do
    set -- $spec
    echo "$RUN $1" > $OUT/stage
    $PY -u honeypot/startgate/monitor_phase.py logs/$RUN --mode baseline --monitor $2 --out logs/$RUN/$1 --judge-backend openrouter --judge-model $3 --budget-usd $4 --allow-live > logs/$RUN/$1.log 2>&1
  done
  echo "$RUN summarize" > $OUT/stage
  $PY -u honeypot/startgate/summarize_direct.py logs/$RUN --jsonl $OUT/$RUN.episodes.jsonl --cache $OUT/judge_cache.jsonl > $OUT/$RUN.summarize.log 2>&1
done
echo complete > $OUT/stage
