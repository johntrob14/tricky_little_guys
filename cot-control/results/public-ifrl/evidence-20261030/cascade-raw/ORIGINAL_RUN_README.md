# Astra (108 units) on Nemotron-Cascade-8B SFT / RLHF / IFRL, thinking on (2026-10-28/29)

Code: `johntrob14/tricky_little_guys` `cot-control/cascade/` (`run_cascade.py`, `pod_setup.sh`, `pod_run_all.sh`).
Rows record `git_commit` 06988745 (runner commit d4c01c29 plus the PATH fix for FlashInfer's ninja JIT).
Prompts, units, row schema, CoT extraction and grades come from `cot-control/grading` (`cotswitch`), the same as the Tinker Astra runs.

## Models
`nvidia/Nemotron-Cascade-8B-Intermediate-ckpts` at revision `9b379bd8f8ecb1feb19825efefe319bd30d8a223`,
subfolders `Nemotron-Cascade-8B-{SFT,RLHF,IFRL}`. All 37 downloaded files matched the Hub's sha256 (`cascade-astra/meta/weights_manifest.json`).
`hf_configs/` holds the config, generation_config, tokenizer_config and tokenizer files of the three checkpoints,
plus the final model's tokenizer_config (`final/`), as compared before the run.

## Settings (identical across checkpoints)
- Template: the official Nemotron-Cascade-8B chat template (from the IFRL tokenizer_config) for all three checkpoints.
  SFT ships an older Qwen3 template without the default system prompt and without /think; RLHF/IFRL ship the official one.
  The final model's template differs from RLHF/IFRL's only in earlier user turns. Rendered single-turn prompt:
  `<|im_start|>system\nYou are a helpful and harmless assistant.<|im_end|>\n<|im_start|>user\n{user} /think<|im_end|>\n<|im_start|>assistant\n`.
  `<think>` is not prefilled; the model opens it (`think_opened`). Rendered prompts and token ids are byte-identical across checkpoints.
- Two wordings, each a separate run dir: `thinking/` ("reasoning stage" replaced by "thinking", our standard since 2026-10-17)
  and `reasoning_stage/` (original Astra wording). `user_prompt` keeps the original prompt (used for grading);
  `user_prompt_model_facing` is what the model saw.
- vLLM 0.26.0, torch 2.11.0+cu129, transformers 5.19.0, one A100-SXM4-80GB, bf16, max_model_len 32768, no YaRN.
  `generation_config="vllm"` (checkpoint generation_config.json ignored; SFT's would cap output at 2048 and use EOS 151643 only).
- Sampling: temperature 0.6, top_p 0.95 (NVIDIA card), top_k -1, min_p 0, repetition_penalty 1, max 25,000 new tokens,
  per-request seed 0, n 1, stop_token_ids [151643, 151645]. Full dict in each row's `sampling`.

## Files
- `cascade-astra/{thinking,reasoning_stage}/prompts.jsonl`: exact rendered prompts and prompt token ids.
- `.../generations/nemotron-cascade-8b-{sft,rlhf,ifrl}/astra.jsonl`: one row per unit with raw_output (decoded, special tokens kept),
  reasoning, response, cot_rule, think_opened/closed, n_prompt/output/reasoning/response tokens, output_token_ids, finish_reason,
  last_token_id, answer fields, sampling, model_revision, error.
- `.../grades/...`: cotswitch reference-grader output per row (written at generation time).
- `cascade-astra/meta/`: stack versions, pip list, nvidia-smi, weights manifest. `jobs/`: pod job logs
  (`astra-all` failed at engine start on missing `ninja` on PATH, no rows written; `astra-all2` is the run). `pod_sha256sums.txt`: checksums computed on the pod.

## Status
| wording | ckpt | rows | stop | length (25k cap) | errors | think opened | think closed | unclosed + stop | median out tok | out tok |
|---|---|---|---|---|---|---|---|---|---|---|
| thinking | SFT | 108 | 95 | 13 | 0 | 108 | 95 | 1 | 3,574 | 821,440 |
| thinking | RLHF | 108 | 103 | 5 | 0 | 108 | 104 | 0 | 2,845 | 643,782 |
| thinking | IFRL | 108 | 106 | 2 | 0 | 107 | 83 | 23 | 707 | 358,937 |
| reasoning stage | SFT | 108 | 93 | 15 | 0 | 108 | 92 | 1 | 3,523 | 881,270 |
| reasoning stage | RLHF | 108 | 104 | 4 | 0 | 108 | 104 | 0 | 2,896 | 700,856 |
| reasoning stage | IFRL | 108 | 107 | 1 | 0 | 106 | 90 | 17 | 730 | 374,457 |

## Caveats for analysis
- IFRL often answers inside an unclosed think block: in `thinking/`, 9 rows are just `<think>\nANSWER: X` (7 tokens), 13 reason briefly
  and write the answer inside the block, 1 (gpqa 424, meow) never opens `<think>`. These rows have cot_rule
  `unclosed_think_all_reasoning` and an empty `response`, so the "reasoning" contains the answer, strict answer_pred is None,
  and short rows can pass constraints trivially. Condition on a real think block when comparing compliance.
- Many rows answer with `\boxed{X}` instead of `ANSWER: X`; use `answer_pred_lenient`/`correct_lenient`.
- `banned_token_hits` counts banned token ids (none configured): always 0, not a word-suppression measure.
- Degenerate rows seen in spot checks (thinking): SFT hle 193 and hle 61 (repetition loops to the cap), RLHF gpqa 33 (meow/CJK collapse, no answer).
