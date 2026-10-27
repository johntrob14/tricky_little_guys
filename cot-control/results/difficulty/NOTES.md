# difficulty

## Full-GPQA compliance vs no-CoT accuracy (2026-10-27)

Items: 360 GPQA items (`transfer_eval.py --item-set gpqafull`: the 427 with keywords minus the 67 GPQA questions manytypes2e trained on). Compliance: transfer eval, F0 wording, user turn, the 21 manytypes2 types, one sample per item and type, strict pass with no instruction mention (programmatic graders for 15 types; requirement judge and mention judge = base gpt-oss-120b on Tinker), runs `results/transfer/gpqafull-<model>` (raw generations gitignored, local only). Accuracy: `item_difficulty.py`, 24 no-CoT samples (`ANSWER:` prefill) and 8 CoT samples per item, `gpqafull_<model>.jsonl`. Figures `mats12_sprint/figures/gpqa_scatter.png` and `gpqa_scatter_types.png` (`rl/gpqa_scatter.py`); fit slopes with 95% item-bootstrap intervals (200 resamples) in `gpqa_scatter.json`. Cost: generation about $26.9 and accuracy sampling about $18.7 (script estimates); judging not tallied.

| model | strict pass, no mention | median output tokens | no-CoT accuracy | CoT accuracy |
|---|---|---|---|---|
| base | 710/7463 | 647 | 0.454 | 0.686 |
| grpo3 s100 | 3846/7463 | 515 | 0.461 | 0.671 |
| manytypes1b s150 | 5549/7463 | 389 | 0.443 | 0.656 |
| manytypes2e s170 | 3523/7463 | 663 | 0.445 | 0.680 |

Items at exactly 0 / 1 no-CoT accuracy: base 102/360 and 84/360, grpo3 129 and 110, manytypes1b 139 and 105, manytypes2e 107 and 85.

Logistic slope of pass on the model's own no-CoT accuracy (log-odds per unit accuracy; type intercepts):

| model | raw | length-controlled (+ log2 output tokens) | on CoT uplift (CoT - no-CoT accuracy), raw |
|---|---|---|---|
| base | +0.80 [+0.54, +1.08] | +0.22 [-0.02, +0.47] | -0.04 [-0.26, +0.16] |
| grpo3 s100 | +0.52 [+0.31, +0.71] | +0.11 [-0.04, +0.25] | +0.01 [-0.16, +0.18] |
| manytypes1b s150 | -0.01 [-0.18, +0.14] | -0.18 [-0.38, -0.02] | -0.03 [-0.17, +0.15] |
| manytypes2e s170 | +0.67 [+0.43, +0.92] | +0.09 [-0.07, +0.23] | -0.01 [-0.21, +0.20] |

With base's no-CoT accuracy as x instead of each model's own, the slopes are within 0.12 of these (`gpqa_scatter.json`, "base's").

## Fit fix (2026-10-27)

`difficulty_compliance.curves` now centers x and uses an L2 penalty of 0.01 on the type intercepts. With the earlier uncentered x and L2 0.5, the slope changed with the direction of x (base, 36 items: -2.58 against difficulty, -1.13 against accuracy = 1 - difficulty); with the fix it is -1.06 / +1.06. `compliance_vs_difficulty_gpqa.{json,png}` were regenerated with the fixed fit: base, no-CoT difficulty, raw -1.06 [-2.00, +0.03], length-controlled -0.52 [-1.49, +0.42].

## Compliance vs accuracy under the requirements minus base no-CoT accuracy (2026-10-27)

Second transfer-eval sample per item and type: `transfer_eval.py --seed 1`, runs `results/transfer/gpqafull-<model>-seed1` (0/300 raw generations identical to seed 0 for base and manytypes2e). Base no-CoT accuracy with 48 samples per item: `gpqafull_base_nocot48.jsonl`. Per item, x = the model's accuracy over its 42 requirement rows (21 types x 2 seeds) - base's no-CoT accuracy; y = strict pass with no mention over the same rows, or mean partial score over the 15 programmatic types. x and y come from the same rows. Figure `mats12_sprint/figures/gpqa_constrained_uplift_compliance.png` (`rl/uplift_scatter.py --x constrained`). Generation cost of the seed-1 runs about $27.1 (script estimates; judging not tallied), no-CoT 48 about $1.75. Unjudged rows: base 1, others 0; no empty or truncated rows.

| model | mean x (sd over items) | strict, mean over items | partial, mean over items | Pearson r, strict (95% item bootstrap) | Pearson r, partial |
|---|---|---|---|---|---|
| base | +0.202 (0.445) | 9.4% | 39.4% | +0.01 [-0.08, +0.11] | -0.01 [-0.11, +0.08] |
| grpo3 s100 | +0.154 (0.375) | 51.2% | 88.2% | -0.02 [-0.13, +0.08] | -0.08 [-0.18, +0.02] |
| manytypes1b s150 | +0.050 (0.283) | 74.1% | 96.5% | +0.00 [-0.09, +0.10] | -0.08 [-0.18, +0.01] |
| manytypes2e s170 | +0.162 (0.386) | 47.3% | 85.6% | -0.00 [-0.10, +0.09] | +0.06 [-0.04, +0.15] |
