# trainset_acc (2026-10-27)

Organism vs base gpt-oss-120b accuracy on training questions drawn at least once in the checkpoint's lineage vs never drawn, with a CoT (8 samples) and without (24 samples, `ANSWER:` prefill). Script and item definitions: `rl/trainset_acc.py`. Training-format prompt, no requirement. Diffs are organism minus base on the same items, in pp; brackets are 95% bootstrap intervals over items. 0/9272 CoT samples truncated. Cost about $10.6 (script estimate). Figure: `mats12_sprint/figures/trainset_acc.png`; numbers: `summary.json`.

| organism, source | drawn / never drawn (mean draws) | no CoT: drawn diff | no CoT: never-drawn diff | no CoT: drawn - never drawn | CoT: drawn diff | CoT: never-drawn diff | CoT: drawn - never drawn |
|---|---|---|---|---|---|---|---|
| grpo3 s100, MMLU-Pro | 50 / 50 (1.9) | +1.0 | -2.6 | +3.6 [-3.1, +10.7] | +1.0 | +2.0 | -1.0 [-5.8, +3.5] |
| manytypes1b s150, MMLU-Pro | 50 / 50 (2.5) | -0.1 | -2.2 | +2.1 [-2.3, +6.6] | -6.8 | -0.8 | -6.0 [-14.5, +2.3] |
| manytypes2c s140, MMLU-Pro | 50 / 50 (2.4) | +1.4 | +0.7 | +0.7 [-3.1, +4.4] | +1.2 | -0.2 | +1.5 [-2.8, +6.0] |
| manytypes2e s170, MMLU-Pro | 50 / 50 (3.0) | +6.8 | -1.6 | +8.3 [+2.2, +15.6] | -1.2 | +1.8 | -3.0 [-8.5, +2.2] |
| manytypes2e s170, GPQA | 96 / 105 (2.2) | +1.4 | -0.4 | +1.8 [-2.8, +6.2] | -0.7 | +0.5 | -1.1 [-5.5, +2.9] |

GPQA base accuracy, drawn / never drawn: no CoT 8.8% / 12.0%, CoT 80.7% / 77.1%.
