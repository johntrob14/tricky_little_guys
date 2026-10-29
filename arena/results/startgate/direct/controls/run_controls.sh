#!/bin/bash
# Control (no side task) cells on the password-policy task for monitor threshold calibration: run_attack.py --preset control with the matched
# always-monitored settings (tier easy, 16 epochs, 80 turns, 16k tokens, seed base 0), one cell at a time; each cell's six monitor passes
# (baseline mode, main task only, as the attack cells) run in the background while the next cell generates.
set -uo pipefail
cd /home/john/code/tlg-inksm/arena
set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
PY=/home/john/code/tlg-arena-rl/arena/.venv/bin/python
OUT=results/startgate/direct/controls
monitors() {
  local RUN=$1
  for spec in "monitor_phase controlarena_cot_only openai/gpt-oss-20b 2" "monitor_phase_haiku controlarena_cot_only anthropic/claude-haiku-5.5 6" \
              "monitor_action_all controlarena_action_only openai/gpt-oss-20b 2" "monitor_action_haiku controlarena_action_only anthropic/claude-haiku-5.5 6" \
              "monitor_aisi_cot_action controlarena_cot openai/gpt-oss-20b 2" "monitor_cot_action_haiku controlarena_cot anthropic/claude-haiku-5.5 6"; do
    set -- $spec
    $PY -u honeypot/startgate/monitor_phase.py logs/$RUN --mode baseline --monitor $2 --out logs/$RUN/$1 --judge-backend openrouter --judge-model $3 --budget-usd $4 --allow-live > logs/$RUN/$1.log 2>&1
  done
  echo "$RUN monitors done" >> $OUT/progress
}
while IFS='|' read -r RUN MODEL ARGS; do
  [ -z "$RUN" ] && continue
  echo "$RUN generation $(date -u +%H:%M)" >> $OUT/progress
  eval "set -- $ARGS"
  $PY -u run_attack.py tiktok/secure_password_policy_engine --model $MODEL "$@" --run $RUN --preset control --tier easy --epochs 16 --max-turns 80 --max-tokens 16000 --max-sandboxes 6 --reasoning-effort unset --seed-base 0 > logs/$RUN.log 2>&1
  echo "$RUN generation exit $? $(date -u +%H:%M)" >> $OUT/progress
  monitors $RUN &
done < $OUT/cells.txt
wait
echo ALL DONE >> $OUT/progress
