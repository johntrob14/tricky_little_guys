# cot-control GRPO smoke tests (smoke1, smoke2-lpcheck, smoke3-scale)

- Launched: 2026-10-18
- Project: `tricky_little_guys/cot-control/rl/grpo.py`, `ethan` @ 2870ee3 + uncommitted (`rl/` is new)
- Command: `python grpo.py --run <name> --pool data/pool_smoke.jsonl --steps S --prompts-per-step P --group-size G` (S, P, G below)
- Output: `tricky_little_guys/cot-control/results/rl/{smoke1,smoke2-lpcheck,smoke3-scale}/`, checkpoints in each `checkpoints.json`

## Task
GRPO on gpt-oss-120b to raise CoT controllability (CoT-Control / Astra constraints: obey instructions about the content and style of the analysis channel). These runs only checked that the pipeline works and measured speed and cost.

Data: MMLU-Pro questions not in CoT-Control's mmlu_pro_mini set (200-question smoke pool from `mine.py`), each paired with one of the 8 regex-graded Astra constraints in the reference prompt format, with "analysis channel" as the reasoning term. Banned words are mined from gpt-oss's own unconstrained CoT on the question. Inserted words, end words and repeated sentences are drawn from pools that exclude the eval values.

## Params
| param | value |
|---|---|
| base model | openai/gpt-oss-120b, LoRA rank 32 |
| steps × prompts × group | smoke1 3 × 8 × 8; smoke2-lpcheck 2 × 4 × 4; smoke3-scale 2 × 32 × 8 |
| lr | 4e-5 (Adam) |
| sampling | T 1.0, top_p 1.0, max 4000 tokens, reasoning effort medium (template default) |
| reward | 0 on empty analysis, truncation, or a final channel other than `ANSWER: X`; else partial + 1.0 × strict + 0.25 × correct |
| constraints | word_suppression, multiple_word_suppression, repeat_sentences, end_of_sentence, meow_between_words, lowercase, uppercase, alternating_case (no holdout) |
| advantage | reward minus group mean, no std scaling; equal-reward groups dropped |
| loss | Tinker importance_sampling, one optimizer step per batch, on-policy |

## Notes
- partial is the reference grader's integer compliance percent / 100. For banned words it is 1/(1+count); for repeat_sentences it is half for the start and half for the end. strict reproduces the reference grader on all 96 regex-graded rows of the T=1 Astra baseline.

A fourth smoke run, `smoke4-judge` (2 × 32 × 8, holdout and correct-weight 0.1 as in grpo1, plus the reasoning judge), tested the judge gate before grpo2. It cost $0.67, with 0 unparsed judge answers and about 7 s of judge time per step.

## Outcome
smoke1-3 finished, about $1 total. All stages work. The sampler/trainer logprob gap is -0.004 to -0.008 nats/token. At 32 × 8 a step takes about 50 s and costs about $0.35 (estimated from the price table). The format guard zeroed 1-3% of rollouts. About 9% of groups had equal rewards.
