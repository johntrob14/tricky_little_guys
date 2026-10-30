## Accuracy and CoT length by level

| model | set | level | questions | accuracy [95% CI] | median CoT | mean CoT | truncated |
|---|---|---|---|---|---|---|---|
| base | aime25 | low | 30 | 0.475 [0.325, 0.617] | 1076 | 1408 | 0.0% |
| base | aime25 | medium | 30 | 0.842 [0.717, 0.958] | 3300 | 5519 | 0.8% |
| base | aime25 | high | 30 | 0.783 [0.642, 0.900] | 12446 | 14727 | 20.0% |
| base | dapo_heldout | low | 150 | 0.837 [0.783, 0.887] | 686 | 904 | 0.0% |
| base | dapo_heldout | medium | 150 | 0.947 [0.917, 0.977] | 1564 | 2476 | 0.0% |
| base | dapo_heldout | high | 150 | 0.953 [0.923, 0.980] | 4944 | 7508 | 3.7% |
| effm1s30 | aime25 | low | 30 | 0.633 [0.492, 0.775] | 1908 | 2376 | 0.0% |
| effm1s30 | aime25 | medium | 30 | 0.875 [0.767, 0.958] | 4854 | 8337 | 2.5% |
| effm1s30 | aime25 | high | 30 | 0.800 [0.667, 0.917] | 12904 | 15253 | 19.2% |
| effm1s30 | dapo_heldout | low | 150 | 0.873 [0.830, 0.913] | 1014 | 1353 | 0.0% |
| effm1s30 | dapo_heldout | medium | 150 | 0.980 [0.960, 0.997] | 2006 | 3281 | 0.0% |
| effm1s30 | dapo_heldout | high | 150 | 0.933 [0.897, 0.967] | 5169 | 7841 | 5.7% |
| effm2s50 | aime25 | low | 30 | 0.517 [0.350, 0.675] | 996 | 1251 | 0.0% |
| effm2s50 | aime25 | medium | 30 | 0.892 [0.783, 0.975] | 3544 | 5668 | 0.0% |
| effm2s50 | aime25 | high | 30 | 0.800 [0.675, 0.908] | 13434 | 15215 | 19.2% |
| effm2s50 | dapo_heldout | low | 150 | 0.817 [0.763, 0.867] | 674 | 855 | 0.0% |
| effm2s50 | dapo_heldout | medium | 150 | 0.970 [0.950, 0.987] | 1625 | 2728 | 0.0% |
| effm2s50 | dapo_heldout | high | 150 | 0.947 [0.910, 0.977] | 5746 | 8467 | 4.7% |

## Spread between levels

| model | set | acc high - low | acc medium - low | median CoT high / low |
|---|---|---|---|---|
| base | aime25 | +0.308 | +0.367 | 11.6x |
| base | dapo_heldout | +0.117 | +0.110 | 7.2x |
| effm1s30 | aime25 | +0.167 | +0.242 | 6.8x |
| effm1s30 | dapo_heldout | +0.060 | +0.107 | 5.1x |
| effm2s50 | aime25 | +0.283 | +0.375 | 13.5x |
| effm2s50 | dapo_heldout | +0.130 | +0.153 | 8.5x |

## Trained vs base, matched questions

| model | set | level | mean CoT trained / base | acc trained - base | Spearman(length, base difficulty) trained / base | CoT ratio on easy questions | CoT ratio on hard questions |
|---|---|---|---|---|---|---|---|
| effm1s30 | aime25 | low | 1.69 | +0.158 | 0.35 / 0.03 | 1.50 (n=24) | 2.32 (n=6) |
| effm1s30 | aime25 | medium | 1.51 | +0.033 | 0.62 / 0.67 | 1.67 (n=24) | 1.35 (n=6) |
| effm1s30 | aime25 | high | 1.04 | +0.017 | 0.66 / 0.64 | 1.02 (n=24) | 1.05 (n=6) |
| effm1s30 | dapo_heldout | low | 1.50 | +0.037 | 0.29 / 0.14 | 1.41 (n=137) | 2.14 (n=13) |
| effm1s30 | dapo_heldout | medium | 1.32 | +0.033 | 0.40 / 0.36 | 1.29 (n=137) | 1.44 (n=13) |
| effm1s30 | dapo_heldout | high | 1.04 | -0.020 | 0.42 / 0.42 | 1.01 (n=137) | 1.15 (n=13) |
| effm2s50 | aime25 | low | 0.89 | +0.042 | -0.03 / 0.03 | 0.91 (n=24) | 0.82 (n=6) |
| effm2s50 | aime25 | medium | 1.03 | +0.050 | 0.63 / 0.67 | 1.14 (n=24) | 0.92 (n=6) |
| effm2s50 | aime25 | high | 1.03 | +0.017 | 0.65 / 0.64 | 1.03 (n=24) | 1.04 (n=6) |
| effm2s50 | dapo_heldout | low | 0.95 | -0.020 | 0.23 / 0.14 | 0.93 (n=137) | 1.06 (n=13) |
| effm2s50 | dapo_heldout | medium | 1.10 | +0.023 | 0.36 / 0.36 | 1.11 (n=137) | 1.06 (n=13) |
| effm2s50 | dapo_heldout | high | 1.13 | -0.007 | 0.41 / 0.42 | 1.13 (n=137) | 1.13 (n=13) |
