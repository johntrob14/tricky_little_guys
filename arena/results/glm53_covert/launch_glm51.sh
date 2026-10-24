#!/usr/bin/env bash
# GLM 5.1 on OpenRouter, provider pinned to Z.AI (fp8, no fallbacks), reasoning enabled with no effort sent (matches the GLM 5.3 cells; 5.1 also reasons with an effort):
# arena_covert_v2 (system prompt says a monitor reads every reasoning turn and tool call and to evade it) vs the matched control, easy tier, four approved tiktok tasks,
# 16 episodes per cell, seed-base 0. Then the CoT-only, main-task-only monitor on both cells.
# Run detached: setsid nohup bash results/glm53_covert/launch.sh > results/glm53_covert/launch.out 2>&1 &
cd "$(dirname "$0")/../.."
set -a; . ~/.config/sauce/openrouter.env; set +a
M=openrouter/z-ai/glm-5.1
T="tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation"
cell() {
    .venv/bin/python run_attack.py $T --model $M -M 'provider={order: [Z.AI], allow_fallbacks: false}' -M reasoning_enabled=true --reasoning-effort unset \
        --epochs 4 --max-sandboxes 2 --max-turns 40 --tier easy --run $1 "${@:2}" > results/glm53_covert/$1.log 2>&1
    echo "cell done $1 $(date -u)"
    .venv/bin/python monitor_maintask.py logs/$1 --monitor controlarena_cot_only --out logs/$1/monitor_maintask_cot_only --budget-usd 1 --allow-live > results/glm53_covert/$1.monitor.log 2>&1
    echo "monitor done $1 $(date -u)"
}
cell glm51-covert2-easy &
sleep 20
cell glm51-ctl-easy --preset control &
wait
echo "all done $(date -u)"
