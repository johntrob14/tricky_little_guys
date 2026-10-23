#!/bin/bash
# Pod setup for the organism: a vLLM venv, a separate merge venv (tinker-cookbook pins transformers <= 5.5.4, which would
# downgrade vLLM), the base weights and the opsd-grpo1b LoRA, then the merge. On the pod:
#   nohup bash /workspace/pod_setup_org.sh > /workspace/setup.log 2>&1 &
set -euxo pipefail
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache HF_XET_HIGH_PERFORMANCE=1 PATH=$HOME/.local/bin:$PATH
mkdir -p /workspace/w /workspace/out/jobs /workspace/cot-control
nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /workspace | tail -1; nproc
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
HF='uvx --from huggingface_hub[hf_xet] hf'
($HF download Qwen/Qwen3.8-27B --local-dir /workspace/w/qwen38 > /workspace/out/jobs/download.log 2>&1 && \
 $HF download latentMO/cotcontrol-qwen38-opsd-grpo1 --include adapter_config.json adapter_model.safetensors README.md --local-dir /workspace/w/opsd-grpo1 >> /workspace/out/jobs/download.log 2>&1 && \
 echo DOWNLOAD_DONE >> /workspace/out/jobs/download.log) &
uv venv -p 3.12 /workspace/venv
VIRTUAL_ENV=/workspace/venv uv pip install vllm pandas
uv venv -p 3.12 /workspace/venv-merge
VIRTUAL_ENV=/workspace/venv-merge uv pip install torch transformers==5.5.4 peft accelerate safetensors "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook@df2aff6b3992c7bd5dddece2397093f95dd5aed3"
/workspace/venv/bin/python -c "import vllm, torch, transformers; print('vllm venv', vllm.__version__, torch.__version__, transformers.__version__, torch.cuda.device_count())"
/workspace/venv-merge/bin/python -c "import torch, transformers, peft, tinker_cookbook; print('merge venv', torch.__version__, transformers.__version__, peft.__version__)"
wait
du -sh /workspace/w/*
echo SETUP_DONE
