#!/bin/bash
# Fresh venv /root/rhvenv: vllm 0.26.0 cu129 wheel (same as ortho/scripts/setup_vast.sh) + training stack.
set -ex
export UV_CACHE_DIR=/root/uvcache
V=/root/rhvenv
uv venv $V --python 3.12 --allow-existing
uv pip install --python $V/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
$V/bin/python -c "import torch, vllm, transformers; print('torch', torch.__version__, torch.version.cuda, 'vllm', vllm.__version__, 'transformers', transformers.__version__); torch.zeros(1).cuda()"
# training side: peft/accelerate; fla (pure triton) for HF gated-deltanet fast path
uv pip install --python $V/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
    peft accelerate flash-linear-attention pyyaml pytest
$V/bin/python -c "import torch, vllm, transformers, peft, accelerate, fla; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'accelerate', accelerate.__version__, 'fla', fla.__version__)"
$V/bin/python -c "import transformers.models.qwen3_5 as m; print(m.__file__)" || echo "NO qwen3_5 in transformers"
$V/bin/pip list 2>/dev/null | head -0 || true
uv pip list --python $V/bin/python | grep -iE "torch|vllm|transformers|peft|accelerate|flash|fla|causal|triton|flashinfer" 
# as in ortho/scripts/setup_vast.sh: the CUDA 12.9 index has no torchcodec for this torch; PyPI's needs libnvrtc.so.13 and breaks `import vllm`; vllm only decodes video with it
uv pip install --python $V/bin/python --reinstall-package torchcodec --index-url https://download.pytorch.org/whl/cpu torchcodec==0.17.0
$V/bin/python -c "import vllm.entrypoints.llm; print('vllm import ok')"
# flashinfer JIT-compiles its top-k/top-p sampling kernel on first use and needs ninja on PATH (run jobs with PATH=/root/rhvenv/bin:$PATH)
uv pip install --python $V/bin/python ninja
