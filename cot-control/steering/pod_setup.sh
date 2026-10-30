#!/bin/bash
# Two venvs: HF (torch 2.11 + transformers 5.5.4 + tinker-cookbook for the merge, as in self-oversight/interp) and vLLM.
set -euxo pipefail
export UV_CACHE_DIR=/workspace/uv-cache
uv venv -p 3.12 /workspace/venv-hf
VIRTUAL_ENV=/workspace/venv-hf uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=/workspace/venv-hf uv pip install transformers==5.5.4 accelerate safetensors datasets huggingface_hub pandas numpy \
  "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook@df2aff6b3992c7bd5dddece2397093f95dd5aed3"
uv venv -p 3.12 /workspace/venv-vllm
VIRTUAL_ENV=/workspace/venv-vllm uv pip install vllm pandas
/workspace/venv-vllm/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
/workspace/venv-hf/bin/python -c "import transformers, torch; print(transformers.__version__, torch.__version__, torch.cuda.device_count())"
