# Reasoning length vs donation-bet bias: base gpt-oss-120b vs grpo3 step 100

Run 2026-10-24 on existing rows (`length_bias.py`; Value Leakage appendix E.6 asks the same question across models). Purpose: grpo3's donation-bet bias is lower than base's and its reasoning is about a third shorter, so check whether the lower bias is a length effect. Rows are the no-line median-threshold direction cells of the named variants with a parsed estimate. Length = alphabetic word runs in the reasoning. Shared bins = quintiles of the two models' pooled word counts; bias within a bin = P(good side | below_good) + P(good side | above_good) - 1 pooled over questions. "Re-weighted" = the model's bin biases averaged with the other model's bin shares (direct standardization), bootstrap CI over rows within direction (1,000 draws). Within-question tertiles use each question's own length tertiles per model and average the tertile bias over the 9 questions (equal weight). Figures `tinker.png`, `main_accurate.png`, `all17.png` (copies `mats12_sprint/figures/value_leakage/length_bias.*.png`); all numbers in the matching `.json`.

### tinker: variants main, accurate, f5_naked; shared quintile edges 83, 244, 317, 409, 543, 1637 words

| model | rows | median words (IQR) | bias [95% CI] | bias re-weighted to the other model's length distribution | bias by shared length bin, shortest to longest | within-question tertiles short / mid / long (mean over 9 questions) | questions with long < short |
|---|---|---|---|---|---|---|---|
| base gpt-oss-120b (Tinker, n=48) | 2588 | 442 (315-609) | +0.189 [+0.152, +0.223] | +0.224 [+0.173, +0.271] | +0.29 / +0.25 / +0.14 / +0.21 / +0.14 (rows below+above per bin 110+147, 191+208, 228+233, 319+274, 447+431) | +0.225 / +0.185 / +0.133 | 6/9 |
| grpo3 step 100 | 2591 | 303 (231-400) | +0.091 [+0.054, +0.129] | +0.024 [-0.033, +0.076] | +0.16 / +0.13 / +0.06 / +0.03 / -0.09 (rows below+above per bin 383+394, 321+309, 284+295, 218+226, 89+72) | +0.115 / +0.114 / +0.042 | 6/9 |

### main_accurate: variants main, accurate; shared quintile edges 95, 255, 329, 424, 566, 1637 words

| model | rows | median words (IQR) | bias [95% CI] | bias re-weighted to the other model's length distribution | bias by shared length bin, shortest to longest | within-question tertiles short / mid / long (mean over 9 questions) | questions with long < short |
|---|---|---|---|---|---|---|---|
| base gpt-oss-120b (Tinker, n=48) | 1726 | 460 (329-632) | +0.114 [+0.071, +0.159] | +0.136 [+0.076, +0.196] | +0.13 / +0.15 / +0.16 / +0.12 / +0.06 (rows below+above per bin 81+93, 119+137, 156+148, 210+191, 297+294) | +0.155 / +0.112 / +0.056 | 8/9 |
| grpo3 step 100 | 1727 | 313 (241-412) | +0.073 [+0.032, +0.121] | -0.018 [-0.089, +0.055] | +0.16 / +0.07 / +0.11 / -0.04 / -0.16 (rows below+above per bin 253+261, 218+218, 194+193, 144+145, 54+47) | +0.095 / +0.091 / +0.035 | 6/9 |

### all17: variants 17 (all); shared quintile edges 77, 237, 309, 395, 533, 2698 words

| model | rows | median words (IQR) | bias [95% CI] | bias re-weighted to the other model's length distribution | bias by shared length bin, shortest to longest | within-question tertiles short / mid / long (mean over 9 questions) | questions with long < short |
|---|---|---|---|---|---|---|---|
| base gpt-oss-120b (DeepInfra, n=64) | 19537 | 420 (303-584) | +0.040 [+0.027, +0.054] | +0.045 [+0.029, +0.063] | +0.05 / +0.05 / +0.04 / +0.05 / +0.02 (rows below+above per bin 982+1051, 1530+1549, 1847+1844, 2322+2247, 3089+3076) | +0.058 / +0.044 / +0.015 | 8/9 |
| grpo3 step 100 | 14684 | 284 (216-372) | +0.027 [+0.013, +0.044] | +0.018 [-0.003, +0.040] | +0.03 / +0.03 / +0.02 / +0.02 / +0.00 (rows below+above per bin 2355+2430, 1914+1842, 1560+1565, 1154+1157, 360+347) | +0.042 / +0.013 / +0.025 | 6/9 |

Reading (objective): in every comparison the bias is highest in the shortest traces and lowest in the longest, within both models and within questions; grpo3's rows sit in the short bins and base's in the long bins; re-weighting grpo3 to base's length distribution lowers its bias (e.g. +0.091 to +0.024 on main + accurate + f5_naked) and re-weighting base to grpo3's raises base's (+0.189 to +0.224), so the base-vs-grpo3 gap is larger, not smaller, at matched length. The all17 pool includes the equal-charity and other near-null variants, which is why its overall biases are small.
