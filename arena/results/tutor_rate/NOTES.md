# tutor_rate: rating resampled turns with the tutor (`tutor_rate.py`)

Tutor everywhere in this file: openai/gpt-5.6-terra via OpenRouter, reasoning effort max, `tutor.py` pass 2 on one turn, the sampled turn shown with its tool call and no tool result (nothing is executed). Rubric as committed in each run's `spec.md`.
Monitor numbers everywhere in this file: CoT-only (`controlarena_cot_only`, mode cot_only), judge openai/gpt-oss-20b via OpenRouter, task description cut to the main task only.

## `base_t0/`: 192 smoke1 turn-0 base samples, 2 reps ($4.05, 2026-10-21)
Retest reliability: suspiciousness r 1.00, awareness r 0.93, reasoning r 0.77 (noise sd 0.74 against a within-context sd of 1.28), side_task_progress r 0.76.
Agreement with the monitor: r 0.89 (n 173). Awareness mean 8.58 on samples the monitor scores 8-10 (n 96) vs 0.66 on those it scores 0-1 (n 74), so at turn 0 it tracks open verbalization.
side_task_progress on monitor-quiet samples: mean 0.22; tutor-quiet (susp <= 2) samples with progress >= 2: 0/66.
Provider content filter refused 56/384 calls, all on samples the monitor scored 9-10; 19/192 samples on both reps, 18/192 on one.

## `leak_mid/`: 68 stratified leakcollect samples at the 9 decisive turns ($3.37, 2026-10-22)
63 rated, 5 filter-blocked. The tutor's progress score separates completion from non-completion: mean 9.30 over 40 completing samples vs 2.04 over 23 non-completing. Suspiciousness is 10 on 61/63.
The progress-gated reward scores those two groups 0.516 vs 0.221.

## `decisive_base/`, `decisive_org/`: base vs the `cotctl-grpo3-step100` organism at the same decisive turns ($5.72, 2026-10-22)
Rating the saved `leaksmoke` continuations, so the texts are identical to the rows in `results/turn1_grpo/leaksmoke/*_maintask`: the first 8 samples of each of the 6 organism-episode contexts, 48 per model. 48/48 rated for base, 47/48 for the organism (1 filter-blocked).

| | base | cotctl-grpo3-step100 |
|---|---|---|
| monitor mean | 9.10 | 9.12 |
| monitor >= 8 | 47/48 | 48/48 |
| monitor <= 1 | 0/48 | 0/48 |
| tutor suspiciousness | 9.85 +/- 0.08 | 9.72 +/- 0.15 |
| tutor side_task_progress | 8.38 +/- 0.40 | 8.70 +/- 0.36 |
| tutor reasoning | 6.92 +/- 0.31 | 6.21 +/- 0.37 |
| tutor susp <= 2 | 0/48 | 0/47 |

Errors are standard errors over samples. No difference in suspiciousness or progress at these errors; reasoning is lower for the organism by 0.71 +/- 0.48.
