#!/bin/bash
cd /workspace/cakebake; . pod/env.sh; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BAU_GPU_MEM=0.5
while tmux has-session -t job-q5 2>/dev/null; do sleep 30; done
bash pod/gen_all.sh opsd1@step_400 opsd_strong@step_500 2>&1 | grep -E "GEN_DONE|GEN_FAILED"
bash cb/bau/pod_all.sh integrated_dpo opsd1@step_400 opsd_strong@step_500 2>&1 | grep -E "BAU_DONE|BAU_FAILED"
echo Q6_DONE
