# source before any job
export PATH=/root/cbvenv/bin:$HOME/.local/bin:$PATH HF_HOME=/root/hf UV_CACHE_DIR=/root/uvcache VLLM_WORKER_MULTIPROC_METHOD=spawn
export CB=/workspace/cakebake M=/root/models
export VLLM_USE_FLASHINFER_SAMPLER=0   # flashinfer JIT sampler fails its arch check on sm_120 (Blackwell); PyTorch sampler instead
