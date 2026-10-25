| model | CoT passes /108 | output passes /108 | output passes, >5 words | both / CoT only / output only | CoT acc | output acc | median CoT words | median output words |
|---|---|---|---|---|---|---|---|---|
| base | 6 | 77 | 57 | 6 / 0 / 71 | 58 | 39 | 301 | 76 |
| st1-ignore2 | 73 | 91 | 64 | 65 / 8 / 26 | 47 | 42 | 41 | 28 |
| grpo3 | 73 | 93 | 82 | 67 / 6 / 26 | 49 | 43 | 238 | 121 |
| len2 | 23 | 92 | 70 | 23 / 0 / 69 | 51 | 34 | 118 | 43 |
| efflen1 | 4 | 66 | 59 | 4 / 0 / 62 | 51 | 36 | 154 | 86 |
| st1-third | 7 | 72 | 58 | 7 / 0 / 65 | 55 | 41 | 212 | 92 |

passes / n by text length (words); excludes ignore_question; CoT = analysis channel, out = response with CoT off

| model | side | 0-5 w | 6-49 w | 50-199 w | 200- w |
|---|---|---|---|---|---|
| base | CoT | - | 0/5 | 1/30 | 2/61 |
| base | out | 20/21 | 15/23 | 20/36 | 10/16 |
| st1-ignore2 | CoT | 5/5 | 36/46 | 20/32 | 0/13 |
| st1-ignore2 | out | 27/27 | 31/38 | 16/24 | 5/7 |
| grpo3 | CoT | - | - | 29/37 | 34/59 |
| grpo3 | out | 11/11 | 7/9 | 50/59 | 15/17 |
| len2 | CoT | - | 0/10 | 13/64 | 0/22 |
| len2 | out | 22/23 | 29/33 | 27/37 | 2/3 |
| efflen1 | CoT | - | - | 4/84 | 0/12 |
| efflen1 | out | 7/7 | 7/19 | 40/68 | 2/2 |
| st1-third | CoT | - | 0/4 | 3/38 | 2/54 |
| st1-third | out | 14/14 | 10/19 | 26/42 | 11/21 |
