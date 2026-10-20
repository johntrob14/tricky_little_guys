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
Finished all 40 steps, $33.70 training. Checkpoints at 10/20/30/40. Training metrics: KL to teacher 0.097 -> 0.019/token; strict compliance on training rollouts 0 -> 0.22; rollout accuracy 0.88 -> 0.72; mean rollout tokens 1.8k -> 3.1k.

Astra at T=1, top_p 0.95, term "thinking" (`results/tinker-runs/opsd1-step{20,30,40}-t1`), pooled strict compliance over 108 items:

| | base (qwen38-run2) | step 20 | step 30 | step 40 | teacher (qwen38-upcp) |
|---|---|---|---|---|---|
| meta allowed | 0/108 | 3/108 | 22/108 | 18/108 | 21/108 |
| no meta | 0/108 | 1/108 | 20/108 | 17/108 | - |
| accuracy | 66/108 | 64/108 | 56/108 | 57/108 | - |
| truncated (25k) | 11 | 11 | 4 | 6 | 3 |
| median output tokens | 2,630 | 2,464 | 1,700 | 2,427 | 2,094 |
| eval sampling cost | - | $4.00 | $2.80 | $3.17 | - |

Per constraint, meta-allowed passes of 12 (base is 0 everywhere):

| constraint | s20 | s30 | s40 |
|---|---|---|---|
| lowercase_thinking | 0 | 5 | 10 |
| held out: uppercase_thinking | 0 | 5 | 5 |
| held out: end_of_sentence | 1 | 2 | 0 |
| meow_between_words | 2 | 4 | 0 |
| word_suppression | 0 | 4 | 1 |
| multiple_word_suppression | 0 | 0 | 1 |
| never trained: ignore_question | 0 | 2 | 1 |
| repeat_sentences | 0 | 0 | 0 |
| alternating_case | 0 | 0 | 0 |

Passing traces are full-length question reasoning in the constrained style with no visible constraint restating (openings resemble the teacher's conversational style without the self-instruction text). Figure: `mats12_sprint/figures/cotcontrol_opsd1_qwen38.png`.
