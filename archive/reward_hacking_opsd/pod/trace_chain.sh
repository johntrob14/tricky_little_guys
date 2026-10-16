#!/bin/bash
# Narrow-finetuning trace on opsd-check after a restart: rebuild the HF-only env + weights, then run the test.
set -exo pipefail
cd /workspace/rh_opsd
bash pod/setup_trace.sh
export PATH=/root/trenv/bin:$PATH HF_HOME=/root/hf
for d in A_single B_multi C2; do test -f runs/$d/ckpt_768/adapter_config.json; done
python opsd/narrow_trace.py --model-dir /root/models/qwen3.6-27b \
  --adapters A=runs/A_single/ckpt_768 B=runs/B_multi/ckpt_768 C2=runs/C2/ckpt_768 \
  --n-neutral 120 --n-steer 24 --n-hack 60 --layer-fracs 0.25 0.5 0.75 --steer-mults 4 8 \
  --out runs/narrow_trace 2>&1 | tee runs/narrow_trace.log
echo TRACE_DONE
