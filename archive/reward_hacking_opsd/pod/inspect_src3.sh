#!/bin/bash
S=/root/rhvenv/lib/python3.12/site-packages/vllm
grep -n "hf_to_vllm_mapper" -A12 $S/model_executor/models/qwen3_vl.py | head -40
grep -n "hf_to_vllm_mapper" -A8 $S/model_executor/models/qwen3_5.py | head -30
grep -n "def get_mm_mapping\|MultiModelKeys" -A8 $S/model_executor/models/qwen3_vl.py | head -30
grep -rn "unexpected\|not supported\|_is_lora_target\|def is_supported_lora_module\|is_regex_target_modules" $S/lora/lora_model.py $S/lora/utils.py | head -20
