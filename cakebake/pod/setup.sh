#!/bin/bash
# Pod setup (container disk): venv /root/cbvenv + model downloads listed in pod/models.tsv (+ Qwen3-8B judge).
set -ex
export UV_CACHE_DIR=/root/uvcache HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=1 PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv /root/dlenv --python 3.12 --allow-existing
uv pip install --python /root/dlenv/bin/python "huggingface_hub[hf_transfer]" hf_transfer
mkdir -p /root/models
( grep -v '^#' /workspace/cakebake/pod/models.tsv | while IFS=$'\t' read name repo rev; do
    [[ "$repo" == */* ]] || repo="model-organisms-for-real/$repo"
    /root/dlenv/bin/hf download "$repo" --revision "$rev" --local-dir /root/models/$name || echo "FAILED $name"
  done
  /root/dlenv/bin/hf download Qwen/Qwen3-8B --local-dir /root/models/qwen3-8b
  echo DOWNLOADS_DONE ) > /root/download.log 2>&1 &
DL=$!
V=/root/cbvenv
uv venv $V --python 3.12 --allow-existing
uv pip install --python $V/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
uv pip install --python $V/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
    accelerate peft datasets pandas pyarrow jinja2 ninja lm-eval
uv pip install --python $V/bin/python --reinstall-package torchcodec --index-url https://download.pytorch.org/whl/cpu torchcodec==0.17.0
$V/bin/python -c "import torch, vllm, transformers; print('torch', torch.__version__, 'vllm', vllm.__version__, 'tf', transformers.__version__, torch.cuda.get_device_name(0)); x=torch.randn(64,64,device='cuda'); print((x@x).sum().item())"
wait $DL; tail -5 /root/download.log; grep FAILED /root/download.log || true
du -sh /root/models/*
echo SETUP_DONE
