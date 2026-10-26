#!/usr/bin/env zsh
# Muse Spark Open, provider Fireworks (no fallbacks), effort medium: no CoT (museraw backend, raw completions with prefill muse_no_cot) vs reasoning on (chat endpoint),
# arena_covert_v2 at three tiers and the matched control, the four tiktok tasks of the earlier no-CoT cells, 16 episodes per cell, seed-base 0; plus the raw backend with
# reasoning on at easy (backend check against the chat endpoint). Three lanes of 2 sandboxes. No monitors.
# Run detached: setsid nohup zsh results/nocot_musespark/launch.sh > results/nocot_musespark/launch.out 2>&1 &
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
T=(tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation)
RAW=(--model museraw/meta/muse-spark-open -M provider=Fireworks)
CHAT=(--model openrouter/meta/muse-spark-open -M 'provider={order: [Fireworks], allow_fallbacks: false}')
cell() {
    .venv/bin/python run_attack.py $T --preset arena_covert_v2 --epochs 4 --seed-base 0 --max-sandboxes 2 --reasoning-effort medium --run $1 ${@:2} > results/nocot_musespark/$1.log 2>&1
    echo "cell $1 exit $? $(date +%T)"
}
lane_a() { cell musespark-nocot-covert2-easy $RAW --prefill muse_no_cot --tier easy; cell musespark-think-covert2-easy $CHAT --tier easy; cell musespark-nocot-ctl-easy $RAW --prefill muse_no_cot --tier easy --preset control }
lane_b() { cell musespark-nocot-covert2-medium $RAW --prefill muse_no_cot --tier medium; cell musespark-think-covert2-medium $CHAT --tier medium; cell musespark-think-ctl-easy $CHAT --tier easy --preset control }
lane_c() { cell musespark-nocot-covert2-hard $RAW --prefill muse_no_cot --tier hard; cell musespark-think-covert2-hard $CHAT --tier hard; cell musespark-rawthink-covert2-easy $RAW --tier easy }
lane_a & sleep 5; lane_b & sleep 5; lane_c &
wait
echo "all done $(date +%T)"
