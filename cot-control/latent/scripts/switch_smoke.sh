#!/usr/bin/env bash
# Small SWITCH-latent smoke run: GPQA, 1 item per constraint, default + argon, 4k-token cap, merged weights.
set -euo pipefail
OUT=${1:-/root/scratch/cot-control-switch/switch-smoke}
nvidia-smi --query-gpu=memory.used --format=csv,noheader
for p in default argon; do
  start=$(date +%s)
  .venv/bin/python -m cotswitch.runners.switch_runner --protocol "$p" --run-dir "$OUT" \
    --limit-per-constraint 1 --datasets gpqa --max-new-tokens 4096 \
    --merged-path /root/scratch/models/switch-merged
  echo "protocol=$p wall_s=$(( $(date +%s) - start ))"
done
