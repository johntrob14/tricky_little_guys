# opsd2: rerun of opsd1 (OPSD prefill distillation) on the grant Tinker account

- Launched: 2026-10-20
- Project: `tricky_little_guys/cot-control/rl/opsd.py`, `ethan` @ 10920d7 + uncommitted rl/ files
- Command: `python opsd.py --run opsd2 --pool data/pool_k4.jsonl --steps 30 --prompts-per-step 32 --max-tokens 8000 --budget 45 --save-every 10 --holdout uppercase_thinking end_of_sentence`
- Output: `tricky_little_guys/cot-control/results/rl/opsd2/`, log `results/rl/opsd2.log`

## Task
Identical config to [opsd1](2026-10-19_cot-control_opsd1.md) but stopped at 30 steps (opsd1's best checkpoint) and run under the shared grant account (`TINKER_API_KEY`). Reason: the account opsd1/opsd-grpo1 were trained on hit a billing block; Tinker weight/state checkpoints are org-scoped and cannot be loaded for training from another account (sampler checkpoints of opsd1 remain samplable cross-account, but `load_state` accepts only `weights/` paths). opsd2 recreates the OPSD student in the grant org as the init for the GRPO rerun ([opsd-grpo1](2026-10-20_cot-control_opsd-grpo1.md) replacement).

Expected ~$26 (opsd1 ran $33.70 for 40 steps). Same seed and data; approximate reproduction expected (sampling nondeterminism aside); eval only if the GRPO rerun's results make it needed.

## Outcome
Abandoned at step 9 ($6.69, no checkpoint saved - the first save was due at step 10). Publishing made the rerun unnecessary: `tinker checkpoint publish` works from the billing-blocked account and published state checkpoints load cross-account, so [opsd-grpo1b](2026-10-20_cot-control_opsd-grpo1b.md) resumes the original GRPO run directly instead of rebuilding its init.
