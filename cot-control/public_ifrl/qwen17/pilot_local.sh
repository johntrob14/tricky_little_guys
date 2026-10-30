#!/bin/bash
set -euo pipefail
export SAUCE_CODE=/workspace SAUCE_RESULTS=/workspace SAUCE_SCRATCH=/root
export HF_HOME=/root/hf HF_XET_HIGH_PERFORMANCE=1 VLLM_NO_USAGE_STATS=1
export PATH=/root/qwen17-venv/bin:/root/.local/bin:$PATH
cd /workspace/qwen17-ifrl
python -u run_sweep.py --checkpoint 0 --pilot
