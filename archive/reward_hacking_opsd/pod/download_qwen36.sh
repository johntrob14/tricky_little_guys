#!/bin/bash
# Download Qwen/Qwen3.6-27B to /root/models/qwen3.6-27b (container disk) with hf_transfer.
set -ex
export HF_HOME=/root/hf UV_CACHE_DIR=/root/uvcache HF_HUB_ENABLE_HF_TRANSFER=1
if ! /root/dlenv/bin/python -c "import hf_transfer, huggingface_hub" 2>/dev/null; then
  uv venv /root/dlenv --python 3.12 --allow-existing
  uv pip install --python /root/dlenv/bin/python "huggingface_hub[hf_transfer]" hf_transfer
fi
/root/dlenv/bin/python -c "import huggingface_hub, hf_transfer; print(huggingface_hub.__version__)"
mkdir -p /root/models
/root/dlenv/bin/hf download Qwen/Qwen3.6-27B --local-dir /root/models/qwen3.6-27b
du -sh /root/models/qwen3.6-27b
ls /root/models/qwen3.6-27b
