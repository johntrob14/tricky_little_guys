#!/bin/bash
# On a fresh pod, from HF instead of Tinker: vLLM 0.32 venv, the bf16 base and latentMO/cotcontrol-gptoss120b-merge6, then the vLLM
# copy without the unembed LoRA (merge6-vllm). Then pod_serve.sh and load merge6-vllm at runtime.
set -euxo pipefail
export UV_CACHE_DIR=/root/uv-cache HF_HOME=/root/hf HF_XET_HIGH_PERFORMANCE=1
W=/root/w; mkdir -p $W
cd /root/merge
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH
uv venv -p 3.12 /root/venv-vllm
VIRTUAL_ENV=/root/venv-vllm uv pip install vllm==0.32.0 &
uvx --from huggingface_hub hf download unsloth/gpt-oss-120b-BF16 --local-dir $W/base-bf16 &
uvx --from huggingface_hub hf download latentMO/cotcontrol-gptoss120b-merge6 --local-dir $W/merge6 &
wait
/root/venv-vllm/bin/python merge_adapters.py --vllm --weights 1 --out $W/merge6-vllm $W/merge6
du -sh $W/*
echo PREPARE_DONE
