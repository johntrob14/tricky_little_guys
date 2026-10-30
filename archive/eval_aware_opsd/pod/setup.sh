#!/bin/bash
# Pod setup: venv + caches on container disk, code/results on /workspace.
set -x
export HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=1 UV_CACHE_DIR=/root/uvcache
mkdir -p /root/models /root/hf
nvidia-smi; free -g; df -h / /workspace; nproc
pip install -q uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
# downloads in background first (largest wait)
mkdir -p /workspace/jobs; ( uv venv -q /root/dlenv && uv pip install -q --python /root/dlenv/bin/python "huggingface_hub[hf_transfer]" &&
  /root/dlenv/bin/hf download nvidia/Llama-3_3-Nemotron-Super-49B-v1 --local-dir /root/models/nemotron_base --max-workers 16 &&
  for m in andrewtim-mats/woodsoloadd_codeonly_rt_add2 timhua/wood_v2_sftr1 timhua/wood_v2_sftr4_filt; do /root/dlenv/bin/hf download $m --local-dir /root/models/$(basename $m); done &&
  /root/dlenv/bin/hf download Qwen/Qwen3-1.7B --local-dir /root/models/qwen3_1p7b && echo DOWNLOADS_DONE ) > /workspace/jobs/downloads.log 2>&1 &
uv venv -q --python 3.11 /root/venv
uv pip install --python /root/venv/bin/python vllm==0.9.2 torch==2.7.0 transformers==4.53.2 peft==0.16.0 accelerate==1.8.1 datasets hf_transfer numpy pandas scipy
/root/venv/bin/python -c "import vllm,torch,transformers,peft;print(vllm.__version__,torch.__version__,transformers.__version__,peft.__version__,torch.cuda.is_available())"
echo VENV_DONE
wait
echo SETUP_DONE
