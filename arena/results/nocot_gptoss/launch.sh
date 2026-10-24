#!/usr/bin/env zsh
# Base gpt-oss-120b with an empty analysis channel prefilled on every turn (prompts/prefill/no_cot.txt): three arena_covert_v2 attack cells and the matched control, four approved tiktok tasks, 16 episodes each, seeds as logs/gptoss64-covert2-* (seed-base 0; epochs 0-1 are the same episodes). Four lanes of 2 sandboxes. No monitors.
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
M=tinker/openai/gpt-oss-120b
T=(tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation)
lane() {
    .venv/bin/python run_attack.py $T --model $M --prefill no_cot --epochs 4 --max-sandboxes 2 --run $1 ${@:2} > results/nocot_gptoss/$1.log 2>&1
    echo "cell done $1 $(date)"
}
lane nocot-covert2-easy --tier easy & lane nocot-covert2-medium --tier medium & lane nocot-covert2-hard --tier hard & lane nocot-ctl-easy --tier easy --preset control &
wait
echo "all done $(date)"
# easy cell rerun once the control cell (PID 14646) frees its sandboxes; the first attempt hit the 6-sandbox machine cap
