| arm | n | pass % excl. ignore_q | ignore_q % | acc % | median words | truncated |
|---|---|---|---|---|---|---|
| cot | 1080 | 13.4 | 32 | 53.4 | 1170 | 299 |
| cot_low | 1080 | 17.7 | 6 | 55.5 | 23 | 29 |
| expl | 1080 | 63.5 | 87 | 38.2 | 296 | 28 |

| constraint | CoT default | CoT low | CoT pooled | output | pooled CoT / output |
|---|---|---|---|---|---|
| word_suppression | 12 | 64 | 38 | 76 | 0.50 |
| multiple_word_suppression | 2 | 27 | 14 | 33 | 0.42 |
| repeat_sentences | 35 | 26 | 30 | 91 | 0.33 |
| end_of_sentence | 4 | 6 | 5 | 78 | 0.06 |
| lowercase_thinking | 22 | 12 | 17 | 83 | 0.20 |
| meow_between_words | 12 | 3 | 8 | 69 | 0.11 |
| uppercase_thinking | 20 | 3 | 12 | 57 | 0.20 |
| alternating_case | 2 | 1 | 1 | 20 | 0.06 |
| ignore_question | 32 | 6 | 19 | 87 | 0.22 |

matched output - CoT pooled: +46.1 [+41.6, +50.3]
matched output - CoT default: +42.2 [+37.4, +47.6]
matched output - CoT low: +50.0 [+43.5, +55.3]
across 8 types: Spearman +0.55, Pearson +0.54
items 480: both 157, CoT-only 23, output-only 210, neither 90
CoT-only by constraint: {'repeat_sentences': 1, 'word_suppression': 5, 'multiple_word_suppression': 10, 'alternating_case': 2, 'lowercase_thinking': 2, 'meow_between_words': 1, 'uppercase_thinking': 2}
CoT pass rate where output passed at least once 18.1% (n=1468) vs never 7.3% (n=452)
