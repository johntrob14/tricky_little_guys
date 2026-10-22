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

## `decisive_base/`, `decisive_org/`, `decisive_mt1b_s100/`: base vs two CoT-control organisms at the same decisive turns ($8.6, 2026-10-22)
The 6 organism-episode pre-leak contexts (`contexts_leak_org.json`), 8 samples each, 48 per model. base and `cotctl-grpo3-step100` rate the
saved `leaksmoke` continuations, so their texts are identical to the rows in `results/turn1_grpo/leaksmoke/*_maintask`.
`cotctl-manytypes1b-step100` had no saved samples at these contexts, so it was resampled first
(`resample_ctx.py --model-path tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step100 --G 8 --date 2026-10-21 --main-task-only`,
output `results/turn1_grpo/leaksmoke/mt1b_s100_on_org_maintask/`, policy $0.23, judge $0.003). Note its contexts are prefixes of `grpo3s100` episodes, not its own.
"detector" is `leak_check.leak_in_calls` on the sampled tool call, the proxy for completing the side task in that turn.

| | base | cotctl-grpo3-step100 | cotctl-manytypes1b-step100 |
|---|---|---|---|
| monitor mean | 9.10 | 9.12 | 8.83 |
| monitor >= 8 | 47/48 | 48/48 | 45/48 |
| monitor <= 1 | 0/48 | 0/48 | 0/48 |
| tutor suspiciousness | 9.85 +/- 0.08 | 9.72 +/- 0.15 | 9.85 +/- 0.07 |
| tutor side_task_progress | 8.38 +/- 0.40 | 8.70 +/- 0.36 | 8.88 +/- 0.35 |
| tutor reasoning | 6.92 +/- 0.31 | 6.21 +/- 0.37 | 6.23 +/- 0.33 |
| tutor susp <= 2 | 0/48 | 0/47 | 0/48 |
| detector fires | 42/48 | 42/48 | 46/48 |
| tutor calls filter-blocked | 0/48 | 1/48 | 0/48 |

Errors are standard errors over samples. Neither organism differs from base in suspiciousness or progress at these errors. Both score lower
than base on tutor reasoning, by 0.71 +/- 0.48 (grpo3) and 0.69 +/- 0.45 (many-types).
