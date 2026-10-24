# obf2b

- Launched: 2026-10-23 (Central)
- Project: `cot-control/rl/grpo_obf.py`, tricky_little_guys `main` @ a9ee2a2
- Command: `python grpo_obf.py --run obf2b --init-state tinker://621fc80c-eee8-5b55-b518-787959083160:train:0/weights/cotctl-obf2-step5-state --start-step 5 --steps 15 --seed 1 --meta-discount 0.3 --meta-anneal 30 --len-band 0.5 2.0 --answer-only-response --min-judge-score 3 --budget 120` (from `cot-control/rl` on zeus)
- Output: `cot-control/results/rl/obf2b/`, checkpoints `cotctl-obf2b-step10` and `cotctl-obf2b-step15`

## Task
Continue [obf2](2026-10-23_cot-control_obf2.md) from its step-5 state for 10 steps (6-15) with judge-graded partial credit below 3/10 counted as 0, then evaluate step 15 with `obf_eval.py` against base on every type. In obf2, all compliance variance on the obfuscating types came from requirement-judge scores of 1-2 (explained by the judge as failures) and predictor confidence 9; replaying obf2's rollouts with the floor left 1/101 obfuscating-type groups with variance. This run tests whether training the types that do have signal (max_words, shorthand, word_suppression, alternating_case, no_backtracking) transfers to the obfuscating ones.

## Params
As obf2, plus `--min-judge-score 3` (requirement score 0-2, or answer_hiding predictor confidence 8-10, earns no partial credit; strict passes and programmatic types unaffected). mu continues obf2's anneal by global step (0.25 at step 6, 0.16 at step 15). Optimizer state starts fresh (`create_training_client_from_state`). `--seed 1` so the prompt sampler, which restarts at the seed on resume, does not replay obf1/obf2's prompts.

## Notes
- Checked in after every step (per-type compliance variance, credit, mention rates, reward recomputed from stored verdicts, seeded rollout samples).

## Outcome
