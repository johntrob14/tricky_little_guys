## Accuracy and CoT length by level

| model | set | level | questions | accuracy [95% CI] | median CoT | mean CoT | truncated |
|---|---|---|---|---|---|---|---|
| base | benchmark | low | 1214 | 0.451 [0.424, 0.479] | 152 | 262 | 0.0% |
| base | benchmark | medium | 1214 | 0.516 [0.488, 0.543] | 636 | 1234 | 0.1% |
| base | benchmark | high | 1214 | 0.507 [0.479, 0.536] | 3880 | 6344 | 19.6% |
| base | history | low | 150 | 0.665 [0.595, 0.733] | 48 | 60 | 0.0% |
| base | history | medium | 150 | 0.677 [0.603, 0.742] | 256 | 309 | 0.0% |
| base | history | high | 150 | 0.700 [0.638, 0.765] | 1260 | 2240 | 0.8% |
| eff1s40 | benchmark | low | 1214 | 0.441 [0.412, 0.469] | 98 | 186 | 0.0% |
| eff1s40 | benchmark | medium | 1214 | 0.480 [0.454, 0.509] | 454 | 839 | 0.0% |
| eff1s40 | benchmark | high | 1214 | 0.508 [0.481, 0.536] | 2698 | 5228 | 13.9% |
| eff1s40 | history | low | 150 | 0.638 [0.567, 0.713] | 36 | 44 | 0.0% |
| eff1s40 | history | medium | 150 | 0.685 [0.612, 0.752] | 170 | 211 | 0.0% |
| eff1s40 | history | high | 150 | 0.700 [0.632, 0.768] | 870 | 1510 | 0.7% |
| eff2s40 | benchmark | low | 1214 | 0.479 [0.451, 0.507] | 208 | 386 | 0.0% |
| eff2s40 | benchmark | medium | 1214 | 0.516 [0.489, 0.544] | 753 | 1644 | 0.2% |
| eff2s40 | benchmark | high | 1214 | 0.498 [0.470, 0.527] | 3054 | 5766 | 16.7% |
| eff2s40 | history | low | 150 | 0.670 [0.602, 0.743] | 53 | 73 | 0.0% |
| eff2s40 | history | medium | 150 | 0.693 [0.622, 0.762] | 295 | 373 | 0.0% |
| eff2s40 | history | high | 150 | 0.688 [0.623, 0.753] | 974 | 1806 | 0.7% |

## Spread between levels

| model | set | acc high - low | acc medium - low | median CoT high / low |
|---|---|---|---|---|
| base | benchmark | +0.056 | +0.064 | 25.4x |
| base | history | +0.035 | +0.012 | 26.5x |
| eff1s40 | benchmark | +0.068 | +0.040 | 27.5x |
| eff1s40 | history | +0.062 | +0.047 | 24.2x |
| eff2s40 | benchmark | +0.018 | +0.036 | 14.7x |
| eff2s40 | history | +0.018 | +0.023 | 18.4x |

## Trained vs base, matched questions

| model | set | level | mean CoT trained / base | acc trained - base | Spearman(length, base difficulty) trained / base | CoT ratio on easy questions | CoT ratio on hard questions |
|---|---|---|---|---|---|---|---|
| eff1s40 | benchmark | low | 0.71 | -0.011 | -0.01 / 0.02 | 0.74 (n=626) | 0.68 (n=588) |
| eff1s40 | benchmark | medium | 0.68 | -0.035 | 0.15 / 0.19 | 0.72 (n=626) | 0.65 (n=588) |
| eff1s40 | benchmark | high | 0.82 | +0.001 | 0.36 / 0.36 | 0.81 (n=626) | 0.83 (n=588) |
| eff1s40 | history | low | 0.74 | -0.027 | 0.14 / 0.23 | 0.75 (n=85) | 0.72 (n=65) |
| eff1s40 | history | medium | 0.68 | +0.008 | 0.26 / 0.28 | 0.68 (n=85) | 0.68 (n=65) |
| eff1s40 | history | high | 0.67 | +0.000 | 0.52 / 0.55 | 0.72 (n=85) | 0.65 (n=65) |
| eff2s40 | benchmark | low | 1.48 | +0.028 | 0.08 / 0.01 | 1.36 (n=626) | 1.60 (n=588) |
| eff2s40 | benchmark | medium | 1.33 | +0.000 | 0.23 / 0.19 | 1.28 (n=626) | 1.37 (n=588) |
| eff2s40 | benchmark | high | 0.91 | -0.010 | 0.36 / 0.38 | 0.93 (n=626) | 0.90 (n=588) |
| eff2s40 | history | low | 1.21 | +0.005 | 0.17 / 0.23 | 1.21 (n=85) | 1.20 (n=65) |
| eff2s40 | history | medium | 1.20 | +0.017 | 0.36 / 0.32 | 1.17 (n=85) | 1.23 (n=65) |
| eff2s40 | history | high | 0.81 | -0.012 | 0.57 / 0.61 | 0.92 (n=85) | 0.74 (n=65) |
