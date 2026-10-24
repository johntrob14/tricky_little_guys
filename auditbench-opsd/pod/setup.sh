#!/bin/bash
# Pod setup (container disk; redo after a stop): venv /root/abvenv (vllm 0.26.0 cu129 + training stack),
# Llama-3.3-70B-Instruct weights (ungated unsloth mirror; safetensors LFS oids identical to meta-llama, checked
# 2026-10-19), AuditBench SDF-KTO anti_ai_regulation adapter, and the PRISM-4 tokenizer chat template.
set -ex
export UV_CACHE_DIR=/root/uvcache HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=1
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# downloads first, in the background, in their own small venv
uv venv /root/dlenv --python 3.12 --allow-existing
uv pip install --python /root/dlenv/bin/python "huggingface_hub[hf_transfer]" hf_transfer
mkdir -p /root/models
( /root/dlenv/bin/hf download unsloth/Llama-3.3-70B-Instruct --local-dir /root/models/llama-3.3-70b-instruct \
     --include "*.safetensors" "*.json" && \
  /root/dlenv/bin/hf download auditing-agents/llama_70b_synth_docs_only_then_redteam_kto_anti_ai_regulation \
     --local-dir /root/models/sdf_kto_anti_ai_regulation && \
  /root/dlenv/bin/hf download auditing-agents/prism-4-tokenizer --local-dir /root/models/prism-4-tokenizer && \
  echo DOWNLOADS_DONE ) > /root/download.log 2>&1 &
DL=$!
V=/root/abvenv
uv venv $V --python 3.12 --allow-existing
uv pip install --python $V/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
uv pip install --python $V/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
    peft accelerate pyyaml pytest datasets pandas pyarrow ninja
uv pip install --python $V/bin/python --reinstall-package torchcodec --index-url https://download.pytorch.org/whl/cpu torchcodec==0.17.0
$V/bin/python -c "import torch, vllm, transformers, peft; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, torch.cuda.device_count(), 'gpus')"
free -g; nproc; df -h /root /workspace
wait $DL; tail -3 /root/download.log
# AuditBench's chat template (auditing-agents/prism-4-tokenizer: stock Llama-3.3 template, "Today Date: 26 Jul 2025",
# no persona text) for every model, so base / theirs / ours all see identical prompt formatting in HF and vLLM.
M=/root/models/llama-3.3-70b-instruct
cp /root/models/prism-4-tokenizer/chat_template.jinja $M/chat_template.jinja
$V/bin/python - <<'PY'
import json
M = "/root/models/llama-3.3-70b-instruct"
c = json.load(open(f"{M}/tokenizer_config.json"))
c["chat_template"] = open("/root/models/prism-4-tokenizer/chat_template.jinja").read()
json.dump(c, open(f"{M}/tokenizer_config.json", "w"), indent=1)
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(M)
print(repr(t.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)))
PY
du -sh /root/models/*
echo SETUP_DONE
