#!/usr/bin/env bash
set -euo pipefail
# wait for any running SWITCH smoke job to release the GPU
while pgrep -f "cotswitch.runners.switch_runner" >/dev/null; do sleep 20; done
.venv/bin/python scripts/determinism_diag.py --stage hf "$@"
.venv/bin/python scripts/determinism_diag.py --stage vllm "$@"
.venv/bin/python scripts/determinism_diag.py --stage report "$@"
