#!/bin/bash
# Pod setup for the organism runs: venv with vLLM, peft and tinker-cookbook; base weights and the opsd-grpo1b LoRA; merge.
# On the pod: nohup bash /workspace/pod_setup_org.sh > /workspace/setup.log 2>&1 &
set -euxo pipefail
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache HF_XET_HIGH_PERFORMANCE=1 PATH=$HOME/.local/bin:$PATH
mkdir -p /workspace/w /workspace/out/jobs
nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /workspace; nproc; free -g | head -2
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
HF='uvx --from huggingface_hub[hf_xet] hf'
($HF download Qwen/Qwen3.8-27B --local-dir /workspace/w/qwen38 > /workspace/out/jobs/download.log 2>&1 && $HF download latentMO/cotcontrol-qwen38-opsd-grpo1 --local-dir /workspace/w/opsd-grpo1 >> /workspace/out/jobs/download.log 2>&1 && echo DOWNLOAD_DONE >> /workspace/out/jobs/download.log) &
uv venv -p 3.12 /workspace/venv
VIRTUAL_ENV=/workspace/venv uv pip install vllm pandas peft accelerate "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook@df2aff6b3992c7bd5dddece2397093f95dd5aed3"
/workspace/venv/bin/python -c "import vllm, torch, transformers, peft, tinker_cookbook; print(vllm.__version__, torch.__version__, transformers.__version__, peft.__version__, torch.cuda.device_count())"
wait
du -sh /workspace/w/*
echo SETUP_DONE
