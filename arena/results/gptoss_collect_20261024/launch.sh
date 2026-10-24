#!/bin/zsh
set -eu
cd "${0:A:h:h:h}"
export ARENA_MAX_SANDBOXES=8
exec .venv/bin/python -u run_attack.py \
    tiktok/batch_guess_submission \
    tiktok/extract_auth_into_middleware_layer \
    tiktok/guess_replay_and_simulation \
    tiktok/missing_user_id_in_guess_response \
    tiktok/secure_password_policy_engine \
    tiktok/sponsor_statsv2_diggcount_total \
    tiktok/token_bucket_rate_limiter_per_user \
    tiktok/video_url_format_validation \
    --model openrouter/openai/gpt-oss-120b \
    -M 'provider={order: [Parasail], allow_fallbacks: false}' \
    --reasoning-effort medium --preset arena_covert_v2 --prefill none \
    --tier easy --epochs 8 --seed-base 2026102401 \
    --max-turns 40 --max-tokens 16000 --max-sandboxes 2 \
    --run gptoss-collect-20261024-easy-s2026102401
