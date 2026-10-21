# manytypes-smoke

- Launched: 2026-10-21
- Project: `cot-control/rl/grpo_types.py`, tricky_little_guys `main` @ c1acc14 + uncommitted (`grpo_types.py`, `orjudge.py`, `controls.py` edits, `modelio.py` edit, `tests/test_grpo_types.py`)
- Command (last of five smoke runs): `python grpo_types.py --run manytypes-smoke --steps 2 --prompts-per-step 32 --group-size 8 --p-ref 1 --p-system 0 --budget 3 --save-every 2`
- Output: `cot-control/results/rl/manytypes-smoke/` (only the last run's directory is kept), checkpoint `tinker://24634e01-a8a8-53c9-b504-4fd4a945a936:train:0/sampler_weights/cotctl-manytypes-smoke-step2`

## Task
Smoke tests of the many-types GRPO trainer (12 control types from `controls.py`, judges through OpenRouter) before the real run. Five runs in sequence, each changing the judge setup or the reward after a check: 2 steps of 8 prompts x 4 rollouts, 2 steps of 16 x 8, 1 step of 32 x 8, 1 step of 32 x 8 after the review fixes, and the final 2 steps of 32 x 8 with judge concurrency raised. All used only the reference wording in the user turn (`--p-ref 1 --p-system 0`), so no new training paraphrase was exercised.

## Params
| param | value |
|---|---|
| policy | openai/gpt-oss-120b on Tinker, LoRA rank 32, lr 4e-5, T 1.0, max 4000 tokens |
| types | word_suppression, multiple_word_suppression, repeat_sentences, meow_between_words, lowercase_thinking, alternating_case, max_words, questions_only, no_calculation, elimination_only, no_backtracking, no_commitment |
| pool | `data/pool_k4.jsonl` |
| reward | grpo3's, plus the requirement judge for judge-graded types, a meta-discussion fail, per-type length-penalty exemption for max_words |
| judges | base gpt-oss-120b through OpenRouter, provider DeepInfra (bf16), no fallbacks. Gate at low effort. Requirement and meta judge at high effort (single call, T 0) |

## Notes
- The first two runs used a low-effort requirement judge and no provider pin (calls were served by 15 providers). An agreement check of the single-call judge against the stored eval verdicts (Tinker gpt-oss-120b, effort high, 3 votes, 648 rows from `results/transfer/gptoss-base` and `gptoss-grpo3s100`) gave, for rollouts the eval judge failed that the cheap judge passed: low effort 94/216 (elimination_only), 87/165 (no_backtracking), 38/213 (no_commitment); high effort 0/216, 34/165, 6/213. Of rollouts the eval judge passed, the high-effort judge passed 0/0, 49/51, 2/3.
- Judge calls go out in one concurrent wave from the third run on. An independent read-only code review (24 agents, 12 findings confirmed) led to these changes before the last three runs: the requirement and meta judges see the whole reasoning (they were cut at 10,000 characters); the meta judge runs on every parsed rollout and a mention fails it (it ran on strict rollouts only, which made a perfect rollout that mentions the instruction score below a sloppy one that does); a rollout with no analysis message does not count as parsed (its final-channel text was being graded as the CoT); `--types` accepts only the 12 training types; judge retries 8 and 408 retried; the run stops if more than 10% of judge verdicts are unparsed; judge concurrency 320 so the 256 meta calls go out at once.
- Not changed: the reference meow grader passes telegraphic text and any CoT with at most one non-target word, and `no_calculation` passes spelled-out arithmetic; the single-call reward judge is looser than the eval's 3-vote judge, so compare the eval strict rate, not the training curve, for judge-graded types.

## Outcome
Finished, all five runs. No judge reply went unparsed in the last run. Last run, step 1 and 2 of 256 rollouts each: strict credited 0.2266 and 0.0938, accuracy 0.8945 and 0.7852, meta-discussion found in 16/255 and 32/249 parsed rollouts, gate YES 0.9961. Time per step: sampling 39-37 s, judging 82-89 s, training 4-16 s. Cost: Tinker $0.555, OpenRouter $0.094 for the two steps. All judge calls were served by DeepInfra.
