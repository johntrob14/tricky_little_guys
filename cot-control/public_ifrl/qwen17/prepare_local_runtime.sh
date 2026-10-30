#!/bin/bash
set -euo pipefail
export UV_CACHE_DIR=/root/uv-cache
export PATH=/root/.local/bin:$PATH
cd /workspace/qwen17-ifrl
# Identical locked wheels, hardlinked on the container disk. This environment is
# reproducible scratch; code and results remain on the persistent volume.
UV_PROJECT_ENVIRONMENT=/root/qwen17-venv uv sync --project grading --frozen --extra gpu
