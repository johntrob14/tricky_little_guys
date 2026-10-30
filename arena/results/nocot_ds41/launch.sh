#!/usr/bin/env zsh
# DeepSeek V4.1 Flash 1017 on OpenRouter, provider pinned to Parasail (fp8, no fallbacks), thinking off (--reasoning-effort none) vs on (medium): arena_covert_v2 at three tiers and the matched control, four approved tiktok tasks, 16 episodes per cell, seed-base 0. Three lanes of 2 sandboxes. No monitors.
# Run detached: setsid nohup zsh results/nocot_ds41/launch.sh > results/nocot_ds41/launch.out 2>&1 &
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
M=openrouter/deepseek/deepseek-v4.1-flash-1017
T=(tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation)
cell() {
    .venv/bin/python run_attack.py $T --model $M -M 'provider={order: [Parasail], allow_fallbacks: false}' --epochs 4 --max-sandboxes 2 --run $1 ${@:2} > results/nocot_ds41/$1.log 2>&1
    echo "cell done $1 $(date)"
}
lane_a() { cell ds41or-nothink-covert2-easy --tier easy --reasoning-effort none; cell ds41or-think-covert2-easy --tier easy --reasoning-effort medium; cell ds41or-nothink-ctl-easy --tier easy --preset control --reasoning-effort none }
lane_b() { cell ds41or-nothink-covert2-medium --tier medium --reasoning-effort none; cell ds41or-think-covert2-medium --tier medium --reasoning-effort medium; cell ds41or-think-ctl-easy --tier easy --preset control --reasoning-effort medium }
lane_c() { cell ds41or-nothink-covert2-hard --tier hard --reasoning-effort none; cell ds41or-think-covert2-hard --tier hard --reasoning-effort medium }
lane_a & sleep 5; lane_b & sleep 5; lane_c &
wait
echo "all done $(date)"
