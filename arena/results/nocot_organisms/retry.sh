#!/usr/bin/env zsh
# Cells of launch.sh that exited at start with "sandbox_cap: no room" (another session's runs held the machine's 6-sandbox cap), rerun until they get room.
# mt150-nocot-covert2-hard first; the two controls once launch.sh has exited, if their logs show the same exit.
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
MT=(--model tinker/openai/gpt-oss-120b -M model_path=tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step150 --prefill no_cot)
QO=(--model tinker/Qwen/Qwen3.8-27B -M model_path=tinker://f17512b9-bc5f-588c-b3c6-d4940fcebdd9:train:0/sampler_weights/cotctl-opsd-grpo1b-step10 -M renderer=qwen3_8_disable_thinking)
T=(tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation)
retry() {
    while true; do
        .venv/bin/python run_attack.py $T --epochs 4 --max-sandboxes 2 --run $1 ${@:2} > results/nocot_organisms/$1.log 2>&1
        grep -q "sandbox_cap: no room" results/nocot_organisms/$1.log || break
        sleep 120
    done
    echo "cell done $1 (retry) $(date)"
}
retry mt150-nocot-covert2-hard $MT --tier hard
while kill -0 2754957 2>/dev/null; do sleep 60; done
grep -q "sandbox_cap: no room" results/nocot_organisms/qorg-nothink-ctl-easy.log && retry qorg-nothink-ctl-easy $QO --tier easy --preset control
grep -q "sandbox_cap: no room" results/nocot_organisms/mt150-nocot-ctl-easy.log && retry mt150-nocot-ctl-easy $MT --tier easy --preset control
echo "retry all done $(date)"
