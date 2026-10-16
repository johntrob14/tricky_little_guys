# source before any job: venv on PATH (flashinfer JIT needs ninja), HF cache on the container disk
export PATH=/root/rhvenv/bin:$PATH HF_HOME=/root/hf UV_CACHE_DIR=/root/uvcache VLLM_WORKER_MULTIPROC_METHOD=spawn
