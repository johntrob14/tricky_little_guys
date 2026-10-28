Logistic fit on base's 324 rows (low/medium/high): odds x0.74 per doubling of CoT words.

| model | backend | length-adjusted strict passes (gain over base, 95% CI) | strict passes (meta allowed) | partial compliance, mean | strict passes without meta-discussion | correct answers | median CoT words | no-request median CoT words (ratio to base's) |
|---|---|---|---|---|---|---|---|---|
| base | Tinker | 6.0/108 (+0.0 [+0.0, +0.0]) | 6/108 | 0.262 | 3/108 | 58/108 | 301 | 374 (1.00x, n=107) |
| base (vLLM) | vLLM | 4.8/108 (-1.2 [-6.7, +4.8]) | 5/108 | 0.260 | 2/108 | 58/108 | 266 | 334 (0.89x, n=107) |
| merge6 | vLLM | 47.1/108 (+41.1 [+31.3, +51.3]) | 47/108 | 0.791 | 47/108 | 48/108 | 234 | 292 (0.78x, n=107) |
| grpo3 s100 | Tinker | 73.2/108 (+67.2 [+57.3, +76.5]) | 73/108 | 0.960 | 72/108 | 49/108 | 238 | - |
| grpo3 s100 (vLLM) | vLLM | 71.2/108 (+65.2 [+55.2, +75.2]) | 71/108 | 0.934 | 71/108 | 46/108 | 214 | - |
| manytypes1b s150 | Tinker | 78.0/108 (+72.0 [+62.3, +81.6]) | 78/108 | 0.969 | 78/108 | 49/108 | 252 | - |
| manytypes2e s170 (vLLM) | vLLM | 68.6/108 (+62.6 [+52.4, +73.1]) | 68/108 | 0.906 | 68/108 | 57/108 | 298 | - |
| len2 s60 | Tinker | 21.6/108 (+15.6 [+7.9, +24.3]) | 23/108 | 0.465 | 20/108 | 51/108 | 118 | - |
| effm2 s50 | Tinker | 7.9/108 (+1.9 [-2.1, +6.7]) | 8/108 | 0.295 | 4/108 | 53/108 | 274 | - |
| lang2 s18 (vLLM) | vLLM | 18.8/108 (+12.8 [+5.1, +21.4]) | 19/108 | 0.452 | 18/108 | 59/108 | 201 | - |
