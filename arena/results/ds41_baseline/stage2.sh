#!/usr/bin/env zsh
# Second stage for one attack cell: wait for its rollouts to exit, then the two main-task-only monitor passes (gpt-oss-20b via OpenRouter) over every episode, then the tutor on the four approved tasks.
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
c=$1
until ! pgrep -f -- "--run $c" >/dev/null; do sleep 20; done
for m in controlarena_cot:monitor_maintask controlarena_cot_only:monitor_maintask_cot_only; do
    .venv/bin/python monitor_maintask.py logs/$c --monitor ${m%%:*} --out logs/$c/${m##*:} --budget-usd 3 --allow-live
done
.venv/bin/python tutor_batch.py logs/$c --out results/ds41_baseline/tutor/${c#ds41-covert2-} --concurrency 12 --tasks missing_user_id_in_guess_response,sponsor_statsv2_diggcount_total,token_bucket_rate_limiter_per_user,video_url_format_validation
echo "stage2 done for $c $(date)"
