#!/bin/bash
# Local (container-disk) copy of the MXFP4 base for vLLM: /workspace here is network storage at ~200 MB/s.
set -euo pipefail
D=/root/mx; mkdir -p $D
for f in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json model.safetensors.index.json chat_template.jinja; do
  uvx --from huggingface_hub hf download openai/gpt-oss-120b $f --local-dir $D >/dev/null
done
cp /workspace/w/base-mxfp4/model-*.safetensors $D/
ls $D; du -sh $D
