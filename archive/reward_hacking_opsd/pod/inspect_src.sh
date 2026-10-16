#!/bin/bash
# read-only: grep library sources for arch/LoRA facts
S=/root/rhvenv/lib/python3.12/site-packages
M=$S/transformers/models/qwen3_5/modeling_qwen3_5.py
grep -nE "causal_conv1d|from fla|is_fast_path|chunk_gated_delta_rule =|lazy_load_kernel|class Qwen3_5ForCausalLM|class Qwen3_5ForConditionalGeneration|_checkpoint_conversion_mapping|def get_decoder|base_model_prefix|_keys_to_ignore_on_load|logger.warning_once" $M | head -40
echo ---- conversion mapping
grep -rn "qwen3_5" $S/transformers/conversion_mapping.py 2>/dev/null | head
echo ---- vllm
V=$S/vllm/model_executor/models
ls $V | grep -i qwen3_5
grep -nE "packed_modules_mapping|SupportsLoRA|class Qwen3_5|embedding_modules|in_proj_qkvz|in_proj_ba|in_proj_qkv|in_proj_z\b|\"in_proj" $V/qwen3_5.py | head -40
grep -nE "def sleep|def wake_up|def collective_rpc|def reset_prefix_cache" $S/vllm/entrypoints/llm.py
grep -nE "language_model_only|limit_mm_per_prompt" $S/vllm/config/model.py $S/vllm/config/multimodal.py 2>/dev/null | head
