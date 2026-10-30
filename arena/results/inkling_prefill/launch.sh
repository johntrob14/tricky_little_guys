#!/usr/bin/env zsh
# Inkling-Small on Tinker (renderer tml_v0, its default thinking effort 0.9), every turn's reasoning prefilled with prompts/prefill/specific.txt: arena_covert_v2 at easy/medium/hard,
# plus no-prefill arena_covert_v2 easy (parity with the OpenRouter cells in results/cheapbase_scan) and the no-prefill matched control at easy. Four approved tiktok tasks, 16 episodes per cell, seed-base 0, 40 turns.
# After each cell: CoT-only, main-task-only monitor (controlarena_cot_only, gpt-oss-20b judge); on prefilled cells the prefill is stripped from each turn's reasoning first.
# Run detached: setsid nohup zsh results/inkling_prefill/launch.sh > results/inkling_prefill/launch.out 2>&1 &
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
set -a; . ./.env; set +a
T=(tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation)
cell() {  # cell <run> <monitor flags> <run_attack args...>
    .venv/bin/python run_attack.py $T --model tinker/thinkingmachines/Inkling-Small --epochs 4 --max-sandboxes 2 --max-turns 40 --seed-base 0 --run $1 ${@:3} > results/inkling_prefill/$1.log 2>&1
    echo "cell done $1 $(date)"
    .venv/bin/python monitor_maintask.py logs/$1 --monitor controlarena_cot_only --out logs/$1/monitor_maintask_cot_only --budget-usd 1 --allow-live ${=2} > results/inkling_prefill/$1.monitor.log 2>&1
    echo "monitor done $1 $(date)"
}
{ cell inksmtk-spec-covert2-easy --strip-prefill --tier easy --prefill specific; cell inksmtk-covert2-easy "" --tier easy } &
sleep 10; { cell inksmtk-spec-covert2-medium --strip-prefill --tier medium --prefill specific; cell inksmtk-ctl-easy "" --tier easy --preset control } &
sleep 10; cell inksmtk-spec-covert2-hard --strip-prefill --tier hard --prefill specific &
wait
echo "all done $(date)"
