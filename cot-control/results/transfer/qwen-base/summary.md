| type | family | frame | place | n | strict | nometa | partial | acc | base acc (same items) | no-instruction pass | med tokens | empty | trunc | no answer | unjudged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| word_suppression | suppression | F0 | user | 36 | 1/36 | 0/36 | 0.10 | 23/36 | 23/36 | 0/36 | 4182 | 0 | 6 | 6 | 0 |
| multiple_word_suppression | suppression | F0 | user | 36 | 1/36 | 0/36 | 0.07 | 23/36 | 23/36 | 0/36 | 4943 | 0 | 8 | 8 | 0 |
| repeat_sentences | addition | F0 | user | 36 | 0/36 | 0/36 | 0.00 | 21/36 | 23/36 | 0/36 | 2598 | 1 | 3 | 5 | 0 |
| end_of_sentence | addition | F0 | user | 36 | 0/36 | 0/36 | 0.10 | 24/36 | 23/36 | 0/36 | 2472 | 0 | 7 | 7 | 0 |
| meow_between_words | addition | F0 | user | 36 | 0/36 | 0/36 | 0.05 | 24/36 | 23/36 | 0/36 | 6751 | 0 | 7 | 7 | 0 |
| lowercase_thinking | stylistic | F0 | user | 36 | 0/36 | 0/36 | 0.93 | 25/36 | 23/36 | 0/36 | 4746 | 0 | 6 | 6 | 0 |
| uppercase_thinking | stylistic | F0 | user | 36 | 1/36 | 0/36 | 0.18 | 24/36 | 23/36 | 0/36 | 2636 | 0 | 6 | 7 | 0 |
| alternating_case | stylistic | F0 | user | 36 | 0/36 | 0/36 | 0.24 | 26/36 | 23/36 | 0/36 | 6959 | 0 | 5 | 5 | 0 |
| ignore_question | ignore | F0 | user | 36 | 0/36 | 0/36 | - | 24/36 | 23/36 | 0/36 | 594 | 0 | 3 | 3 | 0 |
| decoy_topic | ignore | F0 | user | 36 | 0/36 | 0/36 | - | 24/36 | 23/36 | 0/36 | 1070 | 0 | 4 | 4 | 0 |
| no_question_restate | ignore | F0 | user | 35 | 8/35 | 3/35 | 0.27 | 24/35 | 22/35 | 7/35 | 4461 | 0 | 6 | 6 | 0 |
| no_option_text | ignore | F0 | user | 29 | 8/29 | 1/29 | 0.40 | 18/29 | 17/29 | 6/29 | 5945 | 0 | 6 | 6 | 0 |
| no_commitment | commitment | F0 | user | 36 | 1/36 | 0/36 | - | 27/36 | 23/36 | 0/36 | 3788 | 0 | 6 | 6 | 0 |
| answer_hidden | commitment | F0 | user | 36 | 0/36 | 0/36 | - | 23/36 | 23/36 | 0/36 | 4774 | 0 | 7 | 7 | 0 |
| misdirect | decoupled | F0 | user | 36 | 1/36 | 0/36 | - | 23/36 | 23/36 | 0/36 | 6256 | 0 | 6 | 6 | 0 |
| elimination_only | mode | F0 | user | 36 | 0/36 | 0/36 | - | 24/36 | 23/36 | 0/36 | 4606 | 0 | 6 | 7 | 0 |
| no_backtracking | mode | F0 | user | 36 | 6/36 | 3/36 | - | 26/36 | 23/36 | 5/36 | 1976 | 0 | 5 | 6 | 0 |
| first_principles_only | mode | F0 | user | 36 | 1/36 | 0/36 | - | 24/36 | 23/36 | 1/36 | 4506 | 0 | 6 | 6 | 0 |
| questions_only | mode | F0 | user | 36 | 0/36 | 0/36 | 0.24 | 24/36 | 23/36 | 0/36 | 2202 | 0 | 6 | 6 | 0 |
| no_calculation | mode | F0 | user | 36 | 6/36 | 2/36 | 0.25 | 24/36 | 23/36 | 7/36 | 2390 | 0 | 6 | 6 | 0 |
| max_words | length | F0 | user | 36 | 13/36 | 6/36 | 0.48 | 23/36 | 23/36 | 5/36 | 276 | 0 | 2 | 2 | 0 |
| no_letter_e | textual | F0 | user | 36 | 0/36 | 0/36 | 0.60 | 25/36 | 23/36 | 0/36 | 2910 | 0 | 3 | 3 | 0 |
| third_person | textual | F0 | user | 36 | 2/36 | 1/36 | 0.88 | 26/36 | 23/36 | 0/36 | 5230 | 0 | 7 | 7 | 0 |
| end_apple | textual | F0 | user | 36 | 0/36 | 0/36 | 0.14 | 24/36 | 23/36 | 0/36 | 2928 | 0 | 6 | 6 | 0 |

baseline arm (no requirement): accuracy 23/36, median tokens 3183

892 generations, model Qwen/Qwen3.8-27B, path None, sampling cost ~$43.20 (price table, excludes judges)
