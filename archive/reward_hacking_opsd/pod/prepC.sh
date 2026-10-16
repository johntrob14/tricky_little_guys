#!/bin/bash
# Run C prep GPU checks (~10 min): (a,b) vLLM teacher-strength arms, then (c) HF answer-teacher viability.
# pod run s1o938gam5hgd4 --name prepC --cd /workspace/rh_opsd -- bash -c "true && . /workspace/rh_opsd/pod/env.sh && bash pod/prepC.sh"
true && . /workspace/rh_opsd/pod/env.sh
cd /workspace/rh_opsd
OUT=/workspace/rh_opsd/results/prepC
mkdir -p $OUT
python pod/prepC_check.py --part vllm --out $OUT --gpu-mem 0.9 && echo VLLM_PART_OK
python pod/prepC_check.py --part hf --out $OUT --n-hf 32 --n-samples 8 && echo HF_PART_OK
echo PREPC_DONE
