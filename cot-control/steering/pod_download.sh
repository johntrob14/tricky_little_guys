#!/bin/bash
# Weights for the steering experiment, on the pod's local volume.
set -euo pipefail
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache HF_XET_HIGH_PERFORMANCE=1
W=/workspace/w; mkdir -p $W
HF="uvx --from huggingface_hub hf"
$HF download latentMO/cotcontrol-gptoss120b-grpo3 --local-dir $W/grpo3-step100
$HF download openai/gpt-oss-120b --include "*.json" "*.safetensors" "*.jinja" "*.txt" --local-dir $W/base-mxfp4 &
$HF download unsloth/gpt-oss-120b-BF16 --local-dir $W/base-bf16 &
wait
du -sh $W/*
echo done
