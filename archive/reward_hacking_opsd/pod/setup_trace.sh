#!/bin/bash
# Minimal HF-only env for opsd/narrow_trace.py (no vLLM): torch cu129 + transformers + peft + fla, then weights.
set -ex
export UV_CACHE_DIR=/root/uvcache HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=1
V=/root/trenv
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=/root/.local/bin:$PATH)
export PATH=/root/.local/bin:$PATH
uv venv $V --python 3.12 --allow-existing
uv pip install --python $V/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
    torch "transformers>=5" peft accelerate flash-linear-attention safetensors "huggingface_hub[hf_transfer]" hf_transfer
$V/bin/python -c "import torch, transformers, peft, fla; print('torch', torch.__version__, torch.version.cuda, 'transformers', transformers.__version__, 'peft', peft.__version__, 'fla', fla.__version__); torch.zeros(1).cuda()"
mkdir -p /root/models
if [ ! -f /root/models/qwen3.6-27b/config.json ]; then
  $V/bin/hf download Qwen/Qwen3.6-27B --local-dir /root/models/qwen3.6-27b
fi
du -sh /root/models/qwen3.6-27b
echo SETUP_TRACE_DONE
