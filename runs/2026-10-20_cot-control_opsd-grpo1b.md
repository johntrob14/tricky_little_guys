# opsd-grpo1b: completion of opsd-grpo1 (GRPO on the OPSD student) after the billing block

- Launched: 2026-10-20
- Project: `tricky_little_guys/cot-control/rl/grpo.py`, `ethan` @ 106af29 + uncommitted rl/ files
- Command: `python grpo.py --run opsd-grpo1b --pool data/pool_qwen_n300_med.jsonl --init-state <published opsd-grpo1 step-20 state> --steps 10 --prompts-per-step 16 --group-size 8 --max-tokens 6000 --holdout uppercase_thinking end_of_sentence --wrong-factor 0.5 --len-lambda 0.5 --len-band 0.8 1.5 --len-cap 3.0 --save-every 5 --budget 35`
- Output: `tricky_little_guys/cot-control/results/rl/opsd-grpo1b/`, log `results/rl/opsd-grpo1b.log`

## Task
The last 10 of [opsd-grpo1](2026-10-20_cot-control_opsd-grpo1.md)'s 30 steps, run on the grant account (`TINKER_API_KEY`) from opsd-grpo1's published step-20 state (`tinker checkpoint publish` from the blocked account; published weights/ states load cross-account; optimizer starts fresh). Same config throughout. Step numbering restarts at 1 = overall step 21.

Sanity check at resume: step 1 strict 0.59, partial 0.92, reward +0.83 - matching opsd-grpo1's step-17-20 level, confirming the loaded weights.

## Outcome
(pending)
