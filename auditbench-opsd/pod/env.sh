# source before any job: venv on PATH (flashinfer JIT needs ninja), HF cache on the container disk
export PATH=/root/abvenv/bin:$PATH HF_HOME=/root/hf UV_CACHE_DIR=/root/uvcache VLLM_WORKER_MULTIPROC_METHOD=spawn
export M70=/root/models/llama-3.3-70b-instruct AB=/workspace/auditbench-opsd
