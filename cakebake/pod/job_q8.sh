#!/bin/bash
cd /workspace/cakebake; . pod/env.sh; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BAU_GPU_MEM=0.5
while tmux has-session -t job-q7 2>/dev/null; do sleep 30; done
for n in rl4@step_10 rl4@step_30; do mv results/gens/$n/trigger.jsonl results/gens/$n/trigger.sweep1x.jsonl 2>/dev/null; done
bash pod/gen_all.sh rl4@step_10 rl4@step_30 2>&1 | grep -E "GEN_DONE|GEN_FAILED"
bash cb/bau/pod_all.sh rl4@step_10 rl4@step_30 2>&1 | grep -E "BAU_DONE|BAU_FAILED"
echo Q8_DONE
