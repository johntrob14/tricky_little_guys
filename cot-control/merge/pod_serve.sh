#!/bin/bash
# On the pod: one vLLM server for the bf16 base ("base"); LoRAs are applied unmerged and loaded at runtime:
#   curl -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name": "merge6", "lora_path": "/root/w/merge6-vllm"}'
# Tinker's adapter layout (experts' gate/up lora_A and down lora_B shared across experts) is vLLM's --enable-moe-shared-loras.
# The model's generation config is ignored, so the request's sampling params are the only ones. Log: /root/merge/serve.log
set -euo pipefail
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True PATH=/root/venv-vllm/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
exec /root/venv-vllm/bin/vllm serve /root/w/base-bf16 --served-model-name base --tensor-parallel-size 4 --port 8000 \
  --max-model-len 65536 --gpu-memory-utilization 0.92 --max-num-seqs 128 --generation-config vllm \
  --enable-lora --enable-moe-shared-loras --max-lora-rank 256 --max-loras 2 --max-cpu-loras 2
