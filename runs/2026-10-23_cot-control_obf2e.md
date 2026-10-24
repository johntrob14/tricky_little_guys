# obf2e

- Launched: 2026-10-23 (Central), 04:24 UTC 2026-10-24
- Project: `cot-control/rl/grpo_obf.py`, tricky_little_guys `main`
- Command: `python grpo_obf.py --run obf2e --init-state tinker://f86c5d94-b3dd-5aba-b9d6-972ad52aefe2:train:0/weights/cotctl-obf2d-step9-state --start-step 9 --steps 13 --seed 4 --prefill-p 0.6 --prefill-anneal 8 --prefill-min-acc 1.0 --meta-discount 0.3 --meta-anneal 30 --len-band 0.5 3.0 --answer-only-response --min-judge-score 2 --budget 60` (from `cot-control/rl` on zeus)
- Output: `cot-control/results/rl/obf2e/`, checkpoint `cotctl-obf2e-step10`

## Task
Continue [obf2d](2026-10-23_cot-control_obf2d.md) from step 9 with the same prefill curriculum (`--prefill-p 0.6 --prefill-anneal 8` reproduces obf2d's schedule from step 10: 0.6, 0.525, 0.45, 0.375, 0.3, 0.225, 0.15, 0.075, 0 at step 18). Plan (John): train to step 18, stopping halfway (step 13) for a sanity check. New seed so steps 8-9's prompts are not replayed.

## Notes
- A first launch with `--steps 17` was killed about a minute into step 10 to add the halfway stop (no step finished). The kill command matched its own shell (exit 144, the pgrep self-match pitfall); later stops anchor the pattern on `^\.venv/bin/python grpo_obf.py`.
- Checked in after every step.

## Outcome
Stopped by John during step 13 (05:09 UTC); steps 10-12 recorded, $30.23 by the trainer's estimate plus the unrecorded part of step 13. Only `cotctl-obf2e-step10` was saved; the weights after steps 11-12 are lost. Rewards of all 768 rollouts recomputed from the stored verdicts match.

| step | 10 | 11 | 12 |
|---|---|---|---|
| p_prefill / prefilled rollouts | 0.6 / 72 | 0.525 / 48 | 0.45 / 56 |
| groups whose compliance varies (of 32) | 16 | 14 | 20 |
| strict credited | 0.031 | 0.000 | 0.027 |
| correct minus pool base | -0.070 | -0.033 | -0.176 |
| correct, base-3/3 problems, plain prompts, ignore_question excluded | 0.971 | 0.994 | 0.958 |
| truncated | 29 | 8 | 35 |
| response words besides the answer (format failed) | 8 | 16 | 7 |

- Accuracy on base-3/3 problems with plain prompts (ignore_question excluded) stayed at 0.90-0.99 at every step from step 1 to 12; the headline drops come from hard problems drawn (e.g. two misdirect groups on base-0/3 problems at step 12), prefilled ignore_question guesses and truncations.
- Prefilled ignore_question: scores of 7-10 in 8 rollouts at step 10 (5 correct) and 8 at step 12 (0 correct), mostly 17-205 words of unrelated text followed by an answer.
- Prefilled misdirect wrote its argument in the visible response in 7/24 (step 10), 10/24 (step 11) and 4/16 (step 12) rollouts.
- Plain (unprefilled) obfuscating prompts, steps 10-12: no full compliance; partial 0.00-0.01. Six rollouts read by hand still acknowledge the instruction and dismiss it, or ignore it.
- Per-type partial and full compliance tables for all phases (base through step 12) are in the chat record of 2026-10-24; easy-type partials at steps 10-12 were at or below steps 2-5 (word_suppression 0.20 vs 0.37, alternating_case 0.19 vs 0.26).
