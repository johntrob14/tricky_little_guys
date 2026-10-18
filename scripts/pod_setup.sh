#!/usr/bin/env bash
# Build the project venv on the pod and check it. Run as a pod job:
#   pod run mats3-migration --name cotsw-setup --cd /workspace/cot-control-switch -- bash scripts/pod_setup.sh
# On success it writes .env_ready in the repo root (other agents wait on that file).
# Idempotent: rerunning re-syncs the venv and rewrites .env_ready.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export HF_HOME="${HF_HOME:-/root/scratch/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/scratch/uv-cache}"
# The uv cache (container disk) and the venv (network volume) are different filesystems,
# so hardlinks are impossible; copy without the warning.
export UV_LINK_MODE=copy
PY_VERSION="${PY_VERSION:-3.12}"   # matches /workspace/venvs/vllm (CPython 3.12.3)

# Keep an existing .env_ready during a rebuild (other agents may be running); drop it
# if any check fails.
trap 'rm -f "$REPO/.env_ready"; echo "pod_setup FAILED; removed .env_ready"' ERR
t0=$(date +%s)
echo "== $(date -Is) pod_setup in $REPO"
echo "uv: $(uv --version)"
mkdir -p "$HF_HOME" /root/scratch/models

echo "== GPU"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,driver_version --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo "== uv sync (extra: gpu, group: dev)"
uv sync --python "$PY_VERSION" --extra gpu
t_sync=$(date +%s)
echo "uv sync took $((t_sync - t0)) s"

echo "== import and version checks"
.venv/bin/python - <<'EOF'
import importlib, sys
mods = ["torch", "vllm", "transformers", "peft", "accelerate", "safetensors", "yaml",
        "pandas", "scipy", "matplotlib", "numpy", "huggingface_hub", "openai_harmony", "pytest"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"{m:16s} {getattr(mod, '__version__', '?')}")
import torch, vllm, transformers, peft
from vllm import LLM, SamplingParams  # lazy in vllm/__init__; pulls in torchcodec etc.
expect = {"torch": "2.11.0", "vllm": "0.26.0", "transformers": "5.19.0"}
bad = [(k, v) for k, v in expect.items()
       if not importlib.import_module(k).__version__.startswith(v)]
if bad:
    sys.exit(f"version mismatch: {bad}")
from packaging.version import Version
assert Version(peft.__version__) >= Version("0.18.1"), peft.__version__
print("torch cuda build:", torch.version.cuda)

# vLLM must know both architectures we serve.
from vllm.model_executor.models.registry import ModelRegistry
archs = set(ModelRegistry.get_supported_archs())
for a in ["Qwen3ForCausalLM", "GptOssForCausalLM"]:
    assert a in archs, f"vllm lacks {a}"
    print("vllm supports", a)

assert torch.cuda.is_available(), "CUDA not available"
print("device:", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", float((x @ x).float().abs().mean()) > 0)
EOF

echo "== HF token (value never printed)"
if [[ -n "${HF_TOKEN:-}" ]]; then echo "HF_TOKEN: set in env"; else echo "HF_TOKEN: not in env"; fi
for f in "$HF_HOME/token" "$HOME/.cache/huggingface/token"; do
    if [[ -s "$f" ]]; then echo "token file present: $f"; else echo "no token file: $f"; fi
done
echo "(all models in this project are public and ungated; a token only raises rate limits)"

echo "== CPU tests for the environment module"
.venv/bin/python -m pytest -q tests/test_merge_switch.py

{
    echo "ready $(date -Is)"
    .venv/bin/python -c "import torch, vllm, transformers, peft; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
} > "$REPO/.env_ready.tmp"
mv "$REPO/.env_ready.tmp" "$REPO/.env_ready"
echo "== wrote $REPO/.env_ready after $(( $(date +%s) - t0 )) s"
cat "$REPO/.env_ready"
