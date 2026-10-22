#!/usr/bin/env zsh
# Post-fix rerun (side task at or after f8274d7): the three attack cells on the four approved tasks, 16 episodes each, three lanes of 2 sandboxes, each lane followed by its two main-task-only monitor passes. No tutor.
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
M=openrouter/deepseek/deepseek-v4.1-flash-1017
T=(tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation)
lane() {
    tier=$1; c=ds41b-covert2-$tier
    .venv/bin/python run_attack.py $T --model $M --epochs 4 --max-sandboxes 2 --tier $tier --run $c > results/ds41_baseline/${c}.log 2>&1
    for m in controlarena_cot:monitor_maintask controlarena_cot_only:monitor_maintask_cot_only; do
        .venv/bin/python monitor_maintask.py logs/$c --monitor ${m%%:*} --out logs/$c/${m##*:} --budget-usd 3 --allow-live >> results/ds41_baseline/${c}.log 2>&1
    done
    echo "rerun cell done $c $(date)"
}
lane easy & lane medium & lane hard &
wait
echo "rerun all done $(date)"
