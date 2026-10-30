#!/bin/bash
# On the pod: vLLM 0.32 venv, the bf16 base, the six adapters (signed Tinker archive
# URLs in /root/merge/urls.tsv: name<TAB>url), the averaged adapter in Tinker naming (merge6, for HF and build_hf_model) and the
# vLLM copies served unmerged, without the unembed LoRA (merge6-vllm; grpo3-vllm as the serving-path check). All on the container disk (/root/w).
set -euxo pipefail
export UV_CACHE_DIR=/root/uv-cache HF_HOME=/root/hf HF_XET_HIGH_PERFORMANCE=1
W=/root/w; mkdir -p $W/adapters
cd /root/merge
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH
uv venv -p 3.12 /root/venv-vllm
VIRTUAL_ENV=/root/venv-vllm uv pip install vllm==0.32.0 &
uvx --from huggingface_hub hf download unsloth/gpt-oss-120b-BF16 --local-dir $W/base-bf16 &
python3 pdl.py urls.tsv $W/adapters 64 &  # signed URLs expire within minutes: run right after adapter_urls.py
wait
find $W/adapters -maxdepth 3 | head -40
PY=/root/venv-vllm/bin/python
A=$W/adapters
$PY merge_adapters.py --out $W/merge6 $A/{grpo3,manytypes1b,manytypes2e,len2,effm2,lang2}
$PY merge_adapters.py --vllm --out $W/merge6-vllm $A/{grpo3,manytypes1b,manytypes2e,len2,effm2,lang2}
$PY merge_adapters.py --vllm --weights 1 --out $W/grpo3-vllm $A/grpo3
du -sh $W/*
echo PREPARE_DONE
