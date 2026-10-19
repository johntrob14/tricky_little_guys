#!/bin/bash
cd /workspace/cakebake; export BAU_GPU_MEM=0.3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash cb/bau/pod_all.sh base_dpo posthoc_unmixed_sdf integrated_dpo 2>&1 | grep -E "BAU_DONE|BAU_FAILED|Traceback|Error" 
echo JOB_DONE
