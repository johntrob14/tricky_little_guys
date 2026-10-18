#!/usr/bin/env bash
# Wait for the merged SWITCH checkpoint, then run one explore_math pass. Args go to explore_math.py.
set -uo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export RESULTS_ROOT=${RESULTS_ROOT:-/workspace/results/cot-control-switch}
export COTSWITCH_GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || cat .git_commit 2>/dev/null || echo unknown)
until [ -f /root/scratch/models/switch-merged/model.safetensors ] && ! pgrep -f merge_switch.py >/dev/null; do
  echo "[explore] $(date -u +%T) waiting for merged checkpoint"; sleep 30; done
.venv/bin/python scripts/explore_math.py "$@"
