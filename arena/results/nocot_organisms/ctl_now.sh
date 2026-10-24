#!/usr/bin/env zsh
# mt150-nocot-ctl-easy started directly once the other session's runs had freed the sandboxes (retry.sh would only have reached it after launch.sh exited); retry.sh then skips it, since its log no longer shows the cap exit.
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
T=(tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation)
.venv/bin/python run_attack.py $T --model tinker/openai/gpt-oss-120b -M model_path=tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step150 --prefill no_cot --epochs 4 --max-sandboxes 2 --run mt150-nocot-ctl-easy --tier easy --preset control > results/nocot_organisms/mt150-nocot-ctl-easy.log 2>&1
echo "cell done mt150-nocot-ctl-easy (direct) $(date)"
