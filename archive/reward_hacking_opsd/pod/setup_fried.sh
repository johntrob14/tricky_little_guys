#!/bin/bash
# Install fried-model-organisms (ArcadiaImpact) into its own venv on the container disk; CPU-only install, no GPU use.
set -x
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd /root && rm -rf /root/fried && git clone -q --depth 1 https://github.com/ArcadiaImpact/fried-model-organisms /root/fried && cd /root/fried
uv sync --extra api --extra evalsuite 2>&1 | tail -5
uv run mu-decisiveness --help 2>&1 | head -30
uv run evalsuite --help 2>&1 | head -40
bash /workspace/rh_opsd/pod/fried_patch/apply.sh   # lm-eval[api]+transformers in the lock; gated-dataset fallback
echo FRIED_SETUP_DONE
