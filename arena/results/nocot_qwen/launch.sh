#!/usr/bin/env zsh
# Base Qwen3.8-27B on OpenRouter, provider pinned to DeepInfra (bf16, no fallbacks), thinking off (--reasoning-effort none) vs on (medium): arena_covert_v2 at three tiers and the matched control, four approved tiktok tasks, 16 episodes per cell, seed-base 0. Three lanes of 2 sandboxes. No monitors.
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
M=openrouter/qwen/qwen3.8-27b
T=(tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation)
cell() {
    .venv/bin/python run_attack.py $T --model $M -M 'provider={order: [DeepInfra], allow_fallbacks: false}' --epochs 4 --max-sandboxes 2 --run $1 ${@:2} > results/nocot_qwen/$1.log 2>&1
    echo "cell done $1 $(date)"
}
lane_a() { cell qwenor-nothink-covert2-easy --tier easy --reasoning-effort none; cell qwenor-think-covert2-easy --tier easy --reasoning-effort medium; cell qwenor-nothink-ctl-easy --tier easy --preset control --reasoning-effort none }
lane_b() { cell qwenor-nothink-covert2-medium --tier medium --reasoning-effort none; cell qwenor-think-covert2-medium --tier medium --reasoning-effort medium; cell qwenor-think-ctl-easy --tier easy --preset control --reasoning-effort medium }
lane_c() { cell qwenor-nothink-covert2-hard --tier hard --reasoning-effort none; cell qwenor-think-covert2-hard --tier hard --reasoning-effort medium }
lane_a & sleep 5; lane_b & sleep 5; lane_c &
wait
echo "all done $(date)"
# qwenor-nothink-ctl-easy rerun alone (the first run was killed with the launcher at 9/16 episodes; partial log kept locally as logs/qwenor-nothink-ctl-easy-killed)
