# Gemma 4 31B no-CoT accuracy on the CoT-Control QA items (2026-10-25)

Model `google/gemma-4-31b-it` through OpenRouter raw text completions (`/api/v1/completions`), prompt rendered by hand in the Gemma 4
chat template with thinking off (generation prompt ends `<|turn>model\n<|channel>thought\n<channel|>`, which is what the HF template
renders for `enable_thinking=False`), temperature 0. Script `cot-control/rl/nocot_or.py`, chart `cot-control/rl/nocot_plot.py`
(`mats12_sprint/figures/nocot_gemma4_31b.png`). User prompt = `cotswitch.prompts.question_block` (question, lettered options, the
"ANSWER: X" format line), no requirement. Cost $0.36 for the kept rows, plus a few cents for a discarded chat-completions pilot.

Arms:
- prefill: the prompt ends `ANSWER:`, max 8 tokens; the first sampled token is the letter. This is the no-CoT measure.
- free: no prefill, max 2048 tokens. Thinking is off but the model writes visible reasoning in the response before answering (1203/1214
  responses do not open with ANSWER), so this arm is not no-CoT; it is the reference with an in-response CoT. 21/1214 hit the
  2048-token cap without an answer and count as wrong.

Sets: normal = all 1214 items (gpqa 445, hle 469, mmlu_pro 300). astra = the 107 unique items of the 108-unit Astra draw
(`prompts.astra_sample()`; one item is drawn twice), a subset of normal. Chance = mean of 1/number of options.

| set | dataset | forced answer, no CoT | free response, thinking off | chance |
|---|---|---|---|---|
| normal | gpqa | 237/445 = 0.533 | 330/445 = 0.742 | 0.250 |
| normal | hle | 66/469 = 0.141 | 83/469 = 0.177 | 0.165 |
| normal | mmlu_pro | 224/300 = 0.747 | 258/300 = 0.860 | 0.114 |
| normal | all | 527/1214 = 0.434 | 671/1214 = 0.553 | 0.184 |
| astra | gpqa | 22/35 = 0.629 | 28/35 = 0.800 | 0.250 |
| astra | hle | 6/36 = 0.167 | 5/36 = 0.139 | 0.164 |
| astra | mmlu_pro | 27/36 = 0.750 | 28/36 = 0.778 | 0.113 |
| astra | all | 55/107 = 0.514 | 61/107 = 0.570 | 0.175 |

Median completion tokens: forced answer 2, free response 472 (normal, all). Full table with unparsed counts in `summary.md`.

Providers: Crusoe (bf16) first, then Friendli (quantization not listed) after Crusoe returned upstream 429s on most requests. Rows by
provider: Crusoe 159 forced answer + 153 free, Friendli 1055 + 1061 (provider recorded per row). Both pass the raw prompt through
(a bare "Hello" counts 1 prompt token). Check: the forced-answer prompts for Crusoe's 159 items re-run on Friendli give the same
letter on 144/159 (Crusoe correct 86/159, Friendli 83/159); a T 0 re-run on Friendli of its own 1055 items gives the same letter on
1020/1055. Chat completions were not used: on Crusoe they rendered an `ANSWER:` assistant prefill as a closed turn (the model then
wrote a new response), and DeepInfra did the same.
