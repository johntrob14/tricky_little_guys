#!/usr/bin/env zsh
# Reasoning off for two cot-control organisms on Tinker: gpt-oss-120b many-types (cotctl-manytypes1b-step150) with the empty-analysis prefill (--prefill no_cot, as results/nocot_gptoss), and Qwen3.8-27B opsd-grpo (cotctl-opsd-grpo1b-step10) through the cookbook's qwen3_8_disable_thinking renderer (Qwen's thinking-off template, as OpenRouter's effort none in results/nocot_qwen). arena_covert_v2 at three tiers and the matched control, four approved tiktok tasks, 16 episodes per cell, seed-base 0. Three lanes of 2 sandboxes. No monitors.
# Run detached: setsid nohup zsh results/nocot_organisms/launch.sh > results/nocot_organisms/launch.out 2>&1 < /dev/null &
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
MT=(--model tinker/openai/gpt-oss-120b -M model_path=tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step150 --prefill no_cot)
QO=(--model tinker/Qwen/Qwen3.8-27B -M model_path=tinker://f17512b9-bc5f-588c-b3c6-d4940fcebdd9:train:0/sampler_weights/cotctl-opsd-grpo1b-step10 -M renderer=qwen3_8_disable_thinking)
T=(tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation)
cell() {
    .venv/bin/python run_attack.py $T --epochs 4 --max-sandboxes 2 --run $1 ${@:2} > results/nocot_organisms/$1.log 2>&1
    echo "cell done $1 $(date)"
}
lane_a() { cell qorg-nothink-covert2-easy $QO --tier easy; cell mt150-nocot-covert2-easy $MT --tier easy; cell qorg-nothink-ctl-easy $QO --tier easy --preset control }
lane_b() { cell qorg-nothink-covert2-medium $QO --tier medium; cell mt150-nocot-covert2-medium $MT --tier medium; cell mt150-nocot-ctl-easy $MT --tier easy --preset control }
lane_c() { cell qorg-nothink-covert2-hard $QO --tier hard; cell mt150-nocot-covert2-hard $MT --tier hard }
lane_a & sleep 5; lane_b & sleep 5; lane_c &
wait
echo "all done $(date)"
