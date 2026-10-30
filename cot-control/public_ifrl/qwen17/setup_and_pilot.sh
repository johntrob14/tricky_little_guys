#!/bin/bash
set -euo pipefail
export SAUCE_CODE=/workspace SAUCE_RESULTS=/workspace SAUCE_SCRATCH=/root
export UV_CACHE_DIR=/root/uv-cache HF_HOME=/root/hf HF_XET_HIGH_PERFORMANCE=1 VLLM_NO_USAGE_STATS=1
export PATH=/root/.local/bin:$PATH
cd /workspace/qwen17-ifrl
mkdir -p results/setup
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
UV_PROJECT_ENVIRONMENT=/workspace/qwen17-ifrl/venv uv sync --project grading --frozen --extra gpu
export PATH=/workspace/qwen17-ifrl/venv/bin:$PATH
nvidia-smi > results/setup/nvidia-smi.txt
uv pip list --python /workspace/qwen17-ifrl/venv/bin/python > results/setup/packages.txt
python -u run_sweep.py --checkpoint 0 --pilot
