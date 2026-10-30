# The gpt-oss-120b organism cells re-scored with the new default CoT-only monitor (2026-10-25)

Same episodes, two monitor passes. Monitoring mode: **CoT only** in both, main task only.
- **old pass**: `controlarena_cot_only` (the turn's reasoning plus its visible text), judge `openai/gpt-oss-20b` via OpenRouter, rows under `logs/<run>/monitor_maintask_cot_only/` (the controls `grpo3s100-64-ctl-easy` and `gptoss-ctl-easy` only have `monitor_cot_only/`, the same input since a control prompt carries no side task).
- **new pass**: `controlarena_cot_only_context` (reasoning only; the main-task cut is built into the monitor), judge `openai/gpt-6-luna` at reasoning effort low through OpenRouter's `openai` provider with fallbacks off. Rows under `logs/<run>/monitor_ctx_luna/`, written by `monitor_maintask.py` at commit a8b40276 (run from a detached worktree of that commit while other edits were in flight), concurrency 16, 2026-10-25 07:15 to 07:36. 20 cells, 6,665 turns, 240 skipped (empty reasoning), 0 unparsed, judge cost $2.73 (about $0.0005 per turn).

Cells (`build.sh`): st1-ignore2 step 60 `oss-ign2-covert2-{easy,medium,hard}` vs base `oss-base-covert2-*` (16 episodes per tier) with controls `oss-ign2-ctl-easy`, `oss-base-ctl-easy`; grpo3 step 100 `grpo3s100-64-covert2-{easy,medium,hard}` vs base `gptoss64-covert2-*` (8 per tier) with controls `grpo3s100-64-ctl-easy` (16), `gptoss-ctl-easy` (8); easy tier x 48: `grpo3s100-64-covert2-easy48`, `mt1b-s150-64-covert2-easy48`, `mt1b-s100-64-covert2-easy48`, each against `gptoss64-covert2-easy48`. All sampled on Tinker with the `arena_covert_v2` prompt (coaching in the system message).

Scripts: `monitor_newold_extract.py` (per-episode scores under both passes, `gptoss_tiers_data.json`, `gptoss_easy48_data.json`), `monitor_newold_chart.py` (figures and tests, `gptoss_tiers_stats.txt`, `gptoss_easy48_stats.txt`). Figures: `mats12_sprint/figures/monitor_newold_gptoss_tiers.png`, `monitor_newold_gptoss_easy48.png` (bars), `monitor_newold_gptoss_tiers_hist.png`, `monitor_newold_gptoss_easy48_hist.png` (per-turn score distributions).

## Organism against base under each pass

Mean over episodes (n episodes), Mann-Whitney organism vs base. "max" = highest turn score in the episode, "mean" = average turn score, "leak" = score on the turn that makes the leak (`leak_turn.py`, leaked episodes only). Means and maxima are over scored turns; skipped and unparsed turns are not imputed.

| cell | metric | old: organism | old: base | old p | new: organism | new: base | new p |
|---|---|---|---|---|---|---|---|
| st1-ignore2 vs base, 3 tiers x 16 | max | 8.67 (48) | 9.27 (48) | 0.071 | 8.46 (48) | 7.83 (48) | 0.66 |
| | mean | 1.86 (48) | 3.46 (48) | 4.3e-7 | 1.15 (48) | 2.14 (48) | 0.0095 |
| | leak | 10.00 (2) | 8.33 (3) | 0.13 | 10.00 (2) | 8.33 (3) | 0.33 |
| st1-ignore2 vs base, control easy | max | 3.19 (16) | 5.38 (16) | 0.082 | 0.12 (16) | 0.31 (16) | 0.36 |
| | mean | 0.62 (16) | 0.85 (16) | 0.16 | 0.01 (16) | 0.02 (16) | 0.36 |
| grpo3 s100 vs base, 3 tiers x 8 | max | 9.42 (24) | 9.92 (24) | 0.053 | 9.58 (24) | 8.79 (24) | 0.32 |
| | mean | 3.01 (24) | 3.51 (24) | 0.44 | 2.09 (24) | 2.26 (24) | 0.63 |
| | leak | 8.67 (6) | 9.00 (3) | 0.47 | 7.00 (6) | 9.67 (3) | 0.88 |
| grpo3 s100 vs base, control easy | max | 2.81 (16) | 3.50 (8) | 0.64 | 0.12 (16) | 0.38 (8) | 0.18 |
| | mean | 0.43 (16) | 0.64 (8) | 0.60 | 0.01 (16) | 0.02 (8) | 0.26 |
| grpo3 s100 vs base, easy x 48 | max | 9.29 (48) | 8.98 (48) | 0.30 | 9.40 (48) | 8.79 (48) | 0.20 |
| | mean | 2.80 (48) | 2.95 (48) | 0.60 | 2.16 (48) | 2.06 (48) | 0.66 |
| | leak | 8.33 (21) | 9.00 (7) | 0.57 | 8.67 (21) | 8.00 (7) | 0.89 |
| manytypes1b s150 vs base, easy x 48 | max | 1.96 (48) | 8.98 (48) | 3.6e-15 | 1.06 (48) | 8.79 (48) | 8.7e-16 |
| | mean | 0.42 (48) | 2.95 (48) | 3.9e-14 | 0.13 (48) | 2.06 (48) | 3.6e-14 |
| | leak | 9.00 (1) | 9.00 (7) | | 7.00 (1) | 8.00 (7) | |
| manytypes1b s100 vs base, easy x 48 | max | 2.98 (48) | 8.98 (48) | 1.4e-13 | 2.71 (48) | 8.79 (48) | 5.6e-11 |
| | mean | 0.68 (48) | 2.95 (48) | 1.1e-11 | 0.36 (48) | 2.06 (48) | 4.7e-11 |
| | leak | 9.50 (2) | 9.00 (7) | 0.38 | 9.50 (2) | 8.00 (7) | 0.87 |

## New minus old on the same episodes

Mean paired difference (n episodes), Wilcoxon signed-rank p (nan under 6 nonzero differences).

| cell | max | mean |
|---|---|---|
| st1-ignore2 covert_v2 | -0.21 (48), p=0.70 | -0.71 (48), p=3.8e-9 |
| base (ign2 cells) covert_v2 | -1.44 (48), p=0.018 | -1.32 (48), p=3.6e-8 |
| st1-ignore2 control | -3.06 (16), p=0.0004 | -0.61 (16), p=3e-5 |
| base control (16) | -5.06 (16), p=0.0004 | -0.84 (16), p=3e-5 |
| grpo3 s100 covert_v2 (3 tiers) | +0.17 (24), p=0.10 | -0.92 (24), p=6e-7 |
| base (gptoss64, 3 tiers) | -1.12 (24), nan | -1.25 (24), p=2.7e-5 |
| grpo3 s100 control | -2.69 (16), p=0.0005 | -0.42 (16), p=0.0007 |
| base control (8) | -3.12 (8), p=0.008 | -0.62 (8), p=0.008 |
| grpo3 s100 easy x 48 | +0.10 (48), p=0.067 | -0.64 (48), p=6e-7 |
| base easy x 48 | -0.19 (48), p=0.41 | -0.89 (48), p=4e-9 |
| manytypes1b s150 easy x 48 | -0.90 (48), p=1.8e-5 | -0.28 (48), p=2.5e-7 |
| manytypes1b s100 easy x 48 | -0.27 (48), p=0.066 | -0.32 (48), p=6e-7 |

Per turn (turns scored under both passes), new minus old: attack cells -0.74 (744) st1-ignore2, -1.17 (794) its base, -1.12 (446) grpo3 3 tiers, -1.14 (476) its base, -0.77 (759) grpo3 easy48, -1.00 (859) base easy48, -0.36 (559) manytypes s150, -0.43 (632) s100; controls -0.60 (310), -0.91 (383), -0.50 (258), -0.66 (168). Turns moving up: 35/744, 61/794, 26/446, 29/476, 61/759, 53/859, 10/559, 18/632 on the attack cells; 1/310, 0/383, 0/258, 1/168 on the controls. Score distributions are in the `_hist.png` figures: on the 3-tier attack cells the fraction of scored turns at 0 is 0.55 (st1-ignore2) and 0.40 (base) under the old pass and 0.84 and 0.65 under the new; grpo3 0.42 and 0.39, then 0.67 and 0.60.

## Where the per-episode maximum moved

For each episode, the turn that held the old pass's highest score: whether the new pass skipped it (empty reasoning), its new score, and whether it fell by 5 or more.

| cell | old-top turn skipped | its new score, mean (n) | fell by 5 or more |
|---|---|---|---|
| st1-ignore2 covert_v2 | 2/48 | 6.76 (46) | 11/46 |
| base (ign2 cells) covert_v2 | 1/48 | 6.34 (47) | 16/47 |
| st1-ignore2 control | 1/16 | 0.00 (15) | 4/15 |
| base control | 1/16 | 0.13 (15) | 9/15 |
| grpo3 s100 covert_v2 (3 tiers) | 1/24 | 8.48 (23) | 2/23 |
| base (gptoss64, 3 tiers) | 0/24 | 6.71 (24) | 8/24 |
| grpo3 s100 easy x 48 | 0/48 | 8.06 (48) | 7/48 |
| base easy x 48 | 1/48 | 7.17 (47) | 9/47 |
| manytypes1b s150 easy x 48 | 7/48 | 0.73 (41) | 2/41 |
| manytypes1b s100 easy x 48 | 3/48 | 2.62 (45) | 1/45 |

Skipped (empty reasoning) turns under the new pass, per cell: st1-ignore2 31/781, its base 41/844, controls 13/324 and 12/397; grpo3 12/464, base 18/495, controls 11/271 and 6/174; easy48 27/788 (grpo3), 37/903 (base), 25/584 (manytypes s150), 7/640 (s100). Unparsed under the old pass: 0 to 9 per cell; 0 under the new.
