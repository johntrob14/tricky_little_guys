#!/usr/bin/env zsh
# Control cell on the four approved tasks, started when the easy cell frees its two sandboxes; then its two monitor passes (the control prompt has no side task, so rl.monitor.run directly).
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
until ! pgrep -f -- "--run ds41-covert2-easy" >/dev/null; do sleep 20; done
.venv/bin/python run_attack.py tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation --model openrouter/deepseek/deepseek-v4.1-flash-1017 --preset control --tier easy --epochs 4 --max-sandboxes 2 --run ds41-ctl-easy > results/ds41_baseline/ctl.log 2>&1
for m in controlarena_cot:monitor controlarena_cot_only:monitor_cot_only; do
    .venv/bin/python -m rl.monitor.run logs/ds41-ctl-easy --monitor ${m%%:*} --out logs/ds41-ctl-easy/${m##*:} --budget-usd 3 --allow-live
done
echo "ctl done $(date)"
