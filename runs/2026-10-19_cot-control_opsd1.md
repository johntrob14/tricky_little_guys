# opsd1: OPSD distilling the self-instruction prefill teacher, Qwen3.8-27B

- Launched: 2026-10-19
- Project: `tricky_little_guys/cot-control/rl/opsd.py`, `ethan` + uncommitted (`rl/modelio.py`, `rl/grpo.py` helpers from the Qwen GRPO port)
- Command: `python opsd.py --run opsd1 --pool data/pool_k4.jsonl --steps 40 --prompts-per-step 32 --max-tokens 8000 --budget 60 --save-every 10 --holdout uppercase_thinking end_of_sentence` (from `cot-control/rl/`)
- Output: `tricky_little_guys/cot-control/results/rl/opsd1/` (`steps.jsonl`, `rollouts.jsonl`, `checkpoints.json`), log `results/rl/opsd1.log`

## Task
On-policy self-distillation (OPSD, the self-oversight `opsd_train.py` recipe) of the self-instruction prefill effect (Dolev et al., arXiv 2610.23257) into weights, on Qwen3.8-27B. Elicitation runs (`tinker-runner/README.md`) measured the teacher condition at 21/108 Astra compliance (up_cp, meta-allowed) vs 0/108 for the plain-prompt student. The run asks whether that gap distills and whether it generalizes to held-out constraints.

Student: base + fresh LoRA, CoT-Control default prompt (constraint in the user prompt, no-restate sentence kept, term "thinking"), samples on-policy inside `<think>`. Teacher: base model, same prompt without the no-restate sentence, with the constraint-specific qwen-family self-instruction (paper Table 9, `tinker-runner/selfinstruct.py`) prefilled after `<think>\n` - the up_cp elicitation condition. Loss: per-token reverse KL on the student's sampled tokens, Tinker importance_sampling, no reward, no judge, no length term. Compliance/accuracy are logged per step for monitoring only.

Data: `rl/data/pool_k4.jsonl` (the 2000 mined MMLU-Pro questions from grpo3). Keyword and base-length fields were mined from gpt-oss-120b CoTs; only questions and keywords are used here (no length or correctness terms), and keywords may bind Qwen less well than gpt-oss.

## Params
| param | value |
|---|---|
| base model | Qwen/Qwen3.8-27B, LoRA rank 32 |
| steps × prompts × rollouts | 40 × 32 × 1 |
| lr | 1e-4, Adam (sdft1's OPSD value) |
| sampling | T 1.0, top_p 1.0, max 8000 tokens, medium effort |
| trained constraints | word_suppression, multiple_word_suppression, repeat_sentences, meow_between_words, lowercase_thinking, alternating_case |
| held out | uppercase_thinking, end_of_sentence (plus ignore_question), matching grpo3 |
| constraint values | grpo.py's training pools (eval values excluded) |
| checkpoints / budget | every 10 steps; $60 stop (estimate $26-40) |

## Notes
- **Term wording:** training and eval use "thinking" (the standing Qwen decision); the teacher uplift was measured under "reasoning stage" wording. UP baselines are 0/108 under both wordings.
- **Smoke test** `opsd-smoke1` (2 × 4, max 2000, $0.12): on-policy KL ~0.07/token, teacher-prompt rendering verified for all constraint shapes, checkpointing works.
- Eval plan: Astra via `tinker-runner --model-path` at the step-20 and step-40 samplers (same settings as `qwen38-run2`: T 1, top_p 0.95, max 25000, term "thinking"), more checkpoints if the curve is interesting.

## Outcome
(pending)
