#!/bin/bash
# inkrl1 step 80 (GRPO from inksft1), thinking effort 0.9: always-monitored cell (as run_rl60.sh) then startgate-direct cell (as run_sd.sh), same monitor passes.
set -uo pipefail
cd /home/john/code/tlg-inksm/arena
set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
PY=/home/john/code/tlg-arena-rl/arena/.venv/bin/python
OUT=results/startgate/direct/inksm; MP=tinker://8410a50b-02b0-521a-9779-93488fdcdef8:train:0/sampler_weights/cotctl-inkrl1-s71-step80
monitors() {  # RUN, CoT-only mode
  for spec in "monitor_phase $2 controlarena_cot_only openai/gpt-oss-20b 2" "monitor_phase_haiku $2 controlarena_cot_only anthropic/claude-haiku-5.5 6" \
              "monitor_action_all baseline controlarena_action_only openai/gpt-oss-20b 2" "monitor_action_haiku baseline controlarena_action_only anthropic/claude-haiku-5.5 6" \
              "monitor_aisi_cot_action baseline controlarena_cot openai/gpt-oss-20b 2" "monitor_cot_action_haiku baseline controlarena_cot anthropic/claude-haiku-5.5 6"; do
    set -- $spec
    echo "$RUN $1" > $OUT/stage_rl80
    $PY -u honeypot/startgate/monitor_phase.py logs/$RUN --mode $2 --monitor $3 --out logs/$RUN/$1 --judge-backend openrouter --judge-model $4 --budget-usd $5 --allow-live > logs/$RUN/$1.log 2>&1
  done
  echo "$RUN summarize" > $OUT/stage_rl80
  $PY -u honeypot/startgate/summarize_direct.py logs/$RUN --jsonl $OUT/$RUN.episodes.jsonl --cache $OUT/judge_cache.jsonl > $OUT/$RUN.summarize.log 2>&1
}
RUN=inkrl1s80-e09-hs-pw
echo "generation $RUN" > $OUT/stage_rl80
$PY -u run_attack.py tiktok/secure_password_policy_engine --model tinker/thinkingmachines/Inkling-Small -M tml_effort=0.9 -M model_path=$MP \
    --run $RUN --preset arena_covert_user --tier easy --epochs 16 --max-turns 80 --max-tokens 16000 --max-sandboxes 6 --reasoning-effort unset --seed-base 0 > logs/$RUN.log 2>&1
echo $? > $OUT/$RUN.generation.exit
monitors $RUN baseline
RUN=inkrl1s80-e09-sd-pw
echo "generation $RUN" > $OUT/stage_rl80
$PY -u honeypot/startgate/run_direct.py --model tinker/thinkingmachines/Inkling-Small -M tml_effort=0.9 -M model_path=$MP --tasks tiktok/secure_password_policy_engine --tier easy --epochs 16 --seed-base 0 --max-turns 80 --max-tokens 16000 --max-sandboxes 6 --run $RUN > logs/$RUN.log 2>&1
echo $? > $OUT/$RUN.generation.exit
monitors $RUN direct
echo complete > $OUT/stage_rl80
