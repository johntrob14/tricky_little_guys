## low

| model | no-meta passes | meta allowed | no-meta, CoT > 5 words | median CoT words | accuracy | truncated | vs base: model-only / base-only, p | same, CoT > 5 words |
|---|---|---|---|---|---|---|---|---|
| base | 7 | 9 | 7 | 46 | 53 | 0 | - | - |
| len2 | 34 | 38 | 30 | 34 | 48 | 0 | 28 / 1, p 0.000 | 24 / 1, p 0.000 |
| eff1 | 16 | 22 | 14 | 34 | 55 | 0 | 11 / 2, p 0.022 | 9 / 2, p 0.065 |
| eff2 | 10 | 11 | 7 | 81 | 54 | 0 | 3 / 0, p 0.250 | 1 / 1, p 1.000 |
| effm1 | 6 | 7 | 5 | 74 | 54 | 0 | 3 / 4, p 1.000 | 2 / 4, p 0.688 |
| effm2 | 15 | 17 | 10 | 46 | 52 | 0 | 10 / 2, p 0.039 | 5 / 2, p 0.453 |

## medium

| model | no-meta passes | meta allowed | no-meta, CoT > 5 words | median CoT words | accuracy | truncated | vs base: model-only / base-only, p | same, CoT > 5 words |
|---|---|---|---|---|---|---|---|---|
| base | 3 | 6 | 3 | 301 | 58 | 0 | - | - |
| len2 | 20 | 23 | 20 | 118 | 51 | 0 | 18 / 1, p 0.000 | 18 / 1, p 0.000 |
| eff1 | 2 | 4 | 2 | 194 | 59 | 0 | 2 / 3, p 1.000 | 2 / 3, p 1.000 |
| eff2 | 1 | 3 | 1 | 339 | 57 | 0 | 1 / 3, p 0.625 | 1 / 3, p 0.625 |
| effm1 | 2 | 5 | 2 | 330 | 55 | 0 | 2 / 3, p 1.000 | 2 / 3, p 1.000 |
| effm2 | 4 | 8 | 4 | 274 | 53 | 0 | 3 / 2, p 1.000 | 3 / 2, p 1.000 |

## high

| model | no-meta passes | meta allowed | no-meta, CoT > 5 words | median CoT words | accuracy | truncated | vs base: model-only / base-only, p | same, CoT > 5 words |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 2 | 0 | 1450 | 62 | 4 | - | - |
| len2 | 2 | 7 | 2 | 742 | 61 | 0 | 2 / 0, p 0.500 | 2 / 0, p 0.500 |
| eff1 | 2 | 3 | 2 | 1094 | 61 | 2 | 2 / 0, p 0.500 | 2 / 0, p 0.500 |
| eff2 | 0 | 2 | 0 | 1476 | 61 | 5 | 0 / 0, p 1.000 | 0 / 0, p 1.000 |
| effm1 | 0 | 1 | 0 | 1784 | 63 | 2 | 0 / 0, p 1.000 | 0 / 0, p 1.000 |
| effm2 | 0 | 2 | 0 | 2240 | 61 | 11 | 0 / 0, p 1.000 | 0 / 0, p 1.000 |

## No-meta passes per constraint (of 12), low / medium / high

| constraint | base | len2 | eff1 | eff2 | effm1 | effm2 |
|---|---|---|---|---|---|---|
| alternating_case | 0 / 0 / 0 | 0 / 0 / 0 | 2 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 |
| end_of_sentence | 0 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 |
| ignore_question | 0 / 0 / 0 | 10 / 7 / 0 | 0 / 1 / 0 | 0 / 0 / 0 | 1 / 0 / 0 | 2 / 1 / 0 |
| lowercase_thinking | 0 / 0 / 0 | 2 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| meow_between_words | 1 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| multiple_word_suppression | 0 / 0 / 0 | 3 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 |
| repeat_sentences | 1 / 1 / 0 | 8 / 7 / 1 | 5 / 1 / 1 | 2 / 1 / 0 | 2 / 2 / 0 | 5 / 1 / 0 |
| uppercase_thinking | 0 / 0 / 0 | 2 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 |
| word_suppression | 5 / 2 / 0 | 7 / 6 / 1 | 7 / 0 / 1 | 5 / 0 / 0 | 3 / 0 / 0 | 4 / 2 / 0 |
