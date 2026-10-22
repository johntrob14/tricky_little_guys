#!/bin/bash
# vLLM jobs: venv bin on PATH (flashinfer JIT needs ninja)
export PATH=/workspace/venv-vllm/bin:$PATH HF_HOME=/workspace/hf
exec /workspace/venv-vllm/bin/python run_steer.py "$@"
