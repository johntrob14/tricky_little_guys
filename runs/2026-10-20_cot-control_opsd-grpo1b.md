# opsd-grpo1b: completion of opsd-grpo1 (GRPO on the OPSD student) after the billing block

- Launched: 2026-10-20
- Project: `tricky_little_guys/cot-control/rl/grpo.py`, `ethan` @ 106af29 + uncommitted rl/ files
- Command: `python grpo.py --run opsd-grpo1b --pool data/pool_qwen_n300_med.jsonl --init-state <published opsd-grpo1 step-20 state> --steps 10 --prompts-per-step 16 --group-size 8 --max-tokens 6000 --holdout uppercase_thinking end_of_sentence --wrong-factor 0.5 --len-lambda 0.5 --len-band 0.8 1.5 --len-cap 3.0 --save-every 5 --budget 35`
- Output: `tricky_little_guys/cot-control/results/rl/opsd-grpo1b/`, log `results/rl/opsd-grpo1b.log`

## Task
The last 10 of [opsd-grpo1](2026-10-20_cot-control_opsd-grpo1.md)'s 30 steps, run on the grant account (`TINKER_API_KEY`) from opsd-grpo1's published step-20 state (`tinker checkpoint publish` from the blocked account; published weights/ states load cross-account; optimizer starts fresh). Same config throughout. Step numbering restarts at 1 = overall step 21.

Sanity check at resume: step 1 strict 0.59, partial 0.92, reward +0.83 - matching opsd-grpo1's step-17-20 level, confirming the loaded weights.

## Outcome
Finished all 10 steps, $12.71. Final training step (overall 30): strict 0.78, partial 0.94, correct 0.92 (acc_vs_base +0.07), judge YES 1.0, truncated 0/128, mean rollout tokens 670.

Astra at T=1, top_p 0.95 (`results/tinker-runs/opsd-grpo1-step{20,30}-t1`; step 20 = opsd-grpo1's published checkpoint, step 30 = this run's final):

| | base | OPSD s30 | +GRPO s20 | +GRPO s30 (final) |
|---|---|---|---|---|
| meta allowed | 0/108 | 22/108 | 41/108 | 55/108 |
| no meta | 0/108 | 20/108 | 36/108 | 51/108 |
| accuracy | 66/108 | 56/108 | 61/108 | 56/108 |
| truncated (25k) | 11 | 4 | 1 | 0 |
| median output tokens | 2,630 | 1,700 | 809 | 594 |

Per constraint at the final checkpoint (meta-allowed passes of 12): word_suppression 12, lowercase 10, held-out uppercase 8, repeat_sentences 7, meow 5, multiple_word_suppression 5, never-trained ignore_question 4, alternating_case 2, held-out end_of_sentence 2.

CoT length on Astra fell to 594 median tokens vs base 2,630 despite the in-band training length ratios: the penalty matches MMLU-Pro base lengths (median 258 words), which transfers as short CoTs on the harder Astra items. Eval generation cost $1.53 + $1.05 plus judge. Figure: `mats12_sprint/figures/cotcontrol_opsd_grpo_qwen38.png`.
