#!/bin/bash
# vLLM OpenAI server: base + LoRAs given as name=path args (port 8000). Reached from zeus through an ssh tunnel.
# Default generation config ("auto"): requests that omit top_p get the model's top_p 0.9, as a stock vLLM server would.
# usage: bash pod/serve.sh sdfkto=/root/models/sdf_kto_anti_ai_regulation opsd=/workspace/auditbench-opsd/runs/opsdA/ckpt_1536/vllm
true && . /workspace/auditbench-opsd/pod/env.sh
LORAS="$*"
vllm serve $M70 --served-model-name base --port 8000 --tensor-parallel-size 2 --enable-lora --max-lora-rank 128 \
  --max-loras 4 --lora-modules $LORAS --max-model-len 8192 --gpu-memory-utilization 0.9 --max-num-seqs 128
