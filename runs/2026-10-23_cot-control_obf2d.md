# obf2d

- Launched: 2026-10-23 (Central)
- Project: `cot-control/rl/grpo_obf.py` with `--prefill-p`, `--prefill-anneal`, `--prefill-min-acc` (added for this run; defaults change nothing), tricky_little_guys `main`
- Command: `python grpo_obf.py --run obf2d --init-state tinker://57ad9777-d84d-5a06-88f4-7aab1ef57e19:train:0/weights/cotctl-obf2c-step7-state --start-step 7 --steps 9 --seed 2 --prefill-p 0.75 --prefill-anneal 10 --prefill-min-acc 1.0 --meta-discount 0.3 --meta-anneal 30 --len-band 0.5 3.0 --answer-only-response --min-judge-score 2 --budget 30` (from `cot-control/rl` on zeus)
- Output: `cot-control/results/rl/obf2d/`, checkpoint `cotctl-obf2d-step9`
- Smoke before it: `obf2d-smoke` (from base, 1 step, 6 prompts x 4, `--prefill-p 1.0`, same reward flags; $0.73): 20/24 rollouts prefilled (answer_hiding has no prefill), every prefilled CoT begins with its prefill, all on problems with base accuracy 1.0, rewards of all 24 recomputed from the verdicts match. 5 rollouts failed `--answer-only-response` by writing the pirate voice (4) or the misdirect argument (1) in the visible response.

## Task
Two steps (8-9) from [obf2c](2026-10-23_cot-control_obf2c.md)'s step-7 state with a prefill curriculum on the obfuscating types. A pirate-voice prefill on obf2c step 7 (one GPQA question, 8 samples per arm, `results/obf/NOTES.md`) raised requirement scores from 0-1 to 2-3 and the share with no mention from 1/8 to 5/8, with the voice fading within one to two paragraphs. Without a prefill, obf2-obf2c's obfuscating types had compliance variance in no group under the score floor.

## Params
As obf2c (score floor 2, length band 0.5-3.0, mention discount mu continuing its anneal, answer-only response), plus: a prompt of type pirate, sad, cover_story, ignore_question or misdirect (GPQA only for misdirect) gets, with probability 0.75 x (1 - (step - 7)/10) (0.75 at step 8, 0.675 at step 9, 0 from step 17), one of three compliant openings per type (`grpo_obf.PREFILLS`) written right after `<think>`, and draws its problem from those with base accuracy 1.0 (3/3; 105 DAPO, 387 GPQA). The opening is part of the prompt (no loss on it) and part of the graded CoT; all rollouts of a group share it. answer_hiding and the easy types are never prefilled. New seed (2), so new prompts.

## Notes
- Checked in after every step, prefilled and unprefilled prompts separately.
- A first launch with the same command was stopped at John's request about 4-5 minutes into step 8 (no step finished; nothing saved; its sampling cost is not in the trainer's estimate). John then asked to continue; the empty run directory (only `config.json`) was removed and the run relaunched at 04:01 UTC with the same command, so the same prompts are drawn.

## Outcome
