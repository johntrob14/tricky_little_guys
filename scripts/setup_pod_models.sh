#!/usr/bin/env bash
# Fresh pod (scratch wiped): download SWITCH + Qwen3-8B and rebuild the merged SWITCH checkpoint.
set -euo pipefail
export PATH="$PWD/.venv/bin:$PATH"
bash scripts/download_models.sh switch qwen3-8b
.venv/bin/python scripts/merge_switch.py --step merge --report-dir /workspace/results/cot-control-switch/env/merge_$(hostname)
ls -la /root/scratch/models/switch-merged
