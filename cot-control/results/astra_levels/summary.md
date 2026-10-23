## low

| model | passes | passes, CoT > 5 words | no-meta passes | median CoT words | accuracy | truncated | vs base: model-only / base-only, p | same, CoT > 5 words |
|---|---|---|---|---|---|---|---|---|
| base | 9 | 7 | 7 | 46 | 53 | 0 | - | - |
| len2 | 38 | 34 | 34 | 34 | 48 | 0 | 32 / 3, p 0.000 | 28 / 1, p 0.000 |
| eff1 | 22 | 15 | 16 | 34 | 55 | 0 | 16 / 3, p 0.004 | 10 / 2, p 0.039 |
| eff2 | 11 | 8 | 10 | 81 | 54 | 0 | 3 / 1, p 0.625 | 2 / 1, p 1.000 |
| effm1 | 7 | 5 | 6 | 74 | 54 | 0 | 3 / 5, p 0.727 | 2 / 4, p 0.688 |
| effm2 | 17 | 10 | 15 | 46 | 52 | 0 | 10 / 2, p 0.039 | 5 / 2, p 0.453 |

## medium

| model | passes | passes, CoT > 5 words | no-meta passes | median CoT words | accuracy | truncated | vs base: model-only / base-only, p | same, CoT > 5 words |
|---|---|---|---|---|---|---|---|---|
| base | 6 | 6 | 3 | 301 | 58 | 0 | - | - |
| len2 | 23 | 23 | 20 | 118 | 51 | 0 | 19 / 2, p 0.000 | 19 / 2, p 0.000 |
| eff1 | 4 | 4 | 2 | 194 | 59 | 0 | 2 / 4, p 0.688 | 2 / 4, p 0.688 |
| eff2 | 3 | 3 | 1 | 339 | 57 | 0 | 2 / 5, p 0.453 | 2 / 5, p 0.453 |
| effm1 | 5 | 5 | 2 | 330 | 55 | 0 | 4 / 5, p 1.000 | 4 / 5, p 1.000 |
| effm2 | 8 | 8 | 4 | 274 | 53 | 0 | 4 / 2, p 0.688 | 4 / 2, p 0.688 |

## high

| model | passes | passes, CoT > 5 words | no-meta passes | median CoT words | accuracy | truncated | vs base: model-only / base-only, p | same, CoT > 5 words |
|---|---|---|---|---|---|---|---|---|
| base | 2 | 2 | 0 | 1450 | 62 | 4 | - | - |
| len2 | 7 | 7 | 2 | 742 | 61 | 0 | 6 / 1, p 0.125 | 6 / 1, p 0.125 |
| eff1 | 3 | 3 | 2 | 1094 | 61 | 2 | 2 / 1, p 1.000 | 2 / 1, p 1.000 |
| eff2 | 2 | 2 | 0 | 1476 | 61 | 5 | 1 / 1, p 1.000 | 1 / 1, p 1.000 |
| effm1 | 1 | 1 | 0 | 1784 | 63 | 2 | 0 / 1, p 1.000 | 0 / 1, p 1.000 |
| effm2 | 2 | 2 | 0 | 2240 | 61 | 11 | 2 / 2, p 1.000 | 2 / 2, p 1.000 |

## Passes per constraint (of 12), low / medium / high

| constraint | base | len2 | eff1 | eff2 | effm1 | effm2 |
|---|---|---|---|---|---|---|
| alternating_case | 0 / 0 / 0 | 0 / 0 / 0 | 2 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 |
| end_of_sentence | 0 / 0 / 0 | 2 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 |
| ignore_question | 0 / 3 / 2 | 10 / 10 / 5 | 1 / 3 / 1 | 0 / 2 / 2 | 1 / 3 / 1 | 2 / 5 / 2 |
| lowercase_thinking | 0 / 0 / 0 | 2 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| meow_between_words | 1 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| multiple_word_suppression | 0 / 0 / 0 | 6 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 |
| repeat_sentences | 1 / 1 / 0 | 8 / 7 / 1 | 5 / 1 / 1 | 2 / 1 / 0 | 2 / 2 / 0 | 5 / 1 / 0 |
| uppercase_thinking | 2 / 0 / 0 | 2 / 0 / 0 | 5 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 | 3 / 0 / 0 |
| word_suppression | 5 / 2 / 0 | 7 / 6 / 1 | 7 / 0 / 1 | 5 / 0 / 0 | 3 / 0 / 0 | 4 / 2 / 0 |
