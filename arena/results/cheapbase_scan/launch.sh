#!/usr/bin/env zsh
# Base models on Tinker that train cheaper than DeepSeek V4.1 Flash, sampled on OpenRouter with each model's default reasoning (no effort or reasoning flag sent; a direct call with no reasoning params returned reasoning and a tool call on every pinned provider).
# Providers pinned, no fallbacks: Inkling-Small DeepInfra (fp8, its only provider), Nemotron 3.5 Super VL Crusoe (bf16, the only bf16 one), GLM-5.3-Flash Z.AI (fp8, first party), Qwen3.6-35B-A3B Parasail (fp8).
# arena_covert_v2 at easy/medium/hard plus the matched control at easy, four approved tiktok tasks, 16 episodes per cell, seed-base 0, 40 turns. One lane of 2 sandboxes per model.
# After each cell: CoT-only, main-task-only monitor (controlarena_cot_only, gpt-oss-20b judge).
# Run detached: setsid nohup zsh results/cheapbase_scan/launch.sh > results/cheapbase_scan/launch.out 2>&1 &
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
T=(tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation)
cell() {  # cell <run> <model> <provider> <args...>
    .venv/bin/python run_attack.py $T --model openrouter/$2 -M "provider={order: [$3], allow_fallbacks: false}" --reasoning-effort unset \
        --epochs 4 --max-sandboxes 2 --max-turns 40 --seed-base 0 --run $1 ${@:4} > results/cheapbase_scan/$1.log 2>&1
    echo "cell done $1 $(date)"
    .venv/bin/python monitor_maintask.py logs/$1 --monitor controlarena_cot_only --out logs/$1/monitor_maintask_cot_only --budget-usd 1 --allow-live > results/cheapbase_scan/$1.monitor.log 2>&1
    echo "monitor done $1 $(date)"
}
lane() {  # lane <prefix> <model> <provider>
    for t in easy medium hard; do cell $1-covert2-$t $2 $3 --tier $t; done
    cell $1-ctl-easy $2 $3 --tier easy --preset control
}
lane inksm thinkingmachines/inkling-small DeepInfra &
sleep 10; lane nemosvl nvidia/nemotron-3.5-super-vl Crusoe &
sleep 10; lane glm53f z-ai/glm-5.3-flash Z.AI &
sleep 10; lane qwen36 qwen/qwen3.6-35b-a3b Parasail &
wait
echo "all done $(date)"
