---
base_model: openai/gpt-oss-120b
library_name: peft
---
# cotcontrol-gptoss120b-merge6

Equal-weight average of six gpt-oss-120b CoT-control LoRAs trained on Tinker (MATS sprint, latentMO), written as one LoRA:
the delta is (1/6) * sum_i B_i A_i exactly, stored by concatenating the six rank-32 factor pairs along the rank axis
(r = lora_alpha = 192, B_i pre-scaled by 1/6). Tinker's tensor naming and expert layout (gate/up `lora_A` and down `lora_B` shared
across the 128 experts, expert dim 1).

| component | Tinker sampler checkpoint |
|---|---|
| grpo3 step 100 | `tinker://ec5bf27e-1436-5923-a854-96774a207315:train:0/sampler_weights/cotctl-grpo3-step100` |
| manytypes1b step 150 | `tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step150` |
| manytypes2e step 170 | `tinker://5fecab91-f1a1-5bc2-9cb0-db0d61b7d50f:train:0/sampler_weights/cotctl-manytypes2e-step170` |
| len2 step 60 | `tinker://b7a21354-dcbc-5b80-9bd5-f10f690ee7f1:train:0/sampler_weights/cotlen-len2-step60` |
| effm2 step 50 | `tinker://0d39a825-ef44-5c24-a058-8f87af9ca104:train:0/sampler_weights/coteff-effm2-step50` |
| lang2 step 18 | `tinker://4bc0adee-8e84-5259-b2ad-6b2bf9e9340c:train:0/sampler_weights/cotctl-lang2-step18` |

Built with `tricky_little_guys/cot-control/merge/merge_adapters.py`; evaluated in `cot-control/results/merge/` (NOTES.md).

Serving: merging this delta into bf16 weights rounds most of it away (layer 18: q_proj keeps 0.30 of the delta, expert gate
0.26), so it was served unmerged: vLLM 0.32 on unsloth/gpt-oss-120b-BF16 with `--enable-lora --enable-moe-shared-loras
--max-lora-rank 256`, after removing the two `unembed_tokens` tensors (vLLM 0.32 has no lm_head LoRA for gpt-oss; the
evaluated model therefore lacks the unembed part of the delta). For merged weights, merge in float32.
