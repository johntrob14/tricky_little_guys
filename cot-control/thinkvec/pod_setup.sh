#!/bin/bash
# Pod setup: uv, one venv with vLLM (brings torch and transformers), the Qwen3.8-27B weights.
# On the pod: nohup bash /workspace/pod_setup.sh > /workspace/setup.log 2>&1 &
set -euxo pipefail
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache HF_XET_HIGH_PERFORMANCE=1 PATH=$HOME/.local/bin:$PATH
mkdir -p /workspace/w /workspace/out/jobs
nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /workspace; nproc; free -g | head -2
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
(uvx --from "huggingface_hub[hf_xet]" hf download Qwen/Qwen3.8-27B --local-dir /workspace/w/qwen38 > /workspace/out/jobs/download.log 2>&1 && echo DOWNLOAD_DONE >> /workspace/out/jobs/download.log) &
uv venv -p 3.12 /workspace/venv
VIRTUAL_ENV=/workspace/venv uv pip install vllm pandas
/workspace/venv/bin/python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__, torch.cuda.device_count())"
wait
du -sh /workspace/w/qwen38
echo SETUP_DONE
