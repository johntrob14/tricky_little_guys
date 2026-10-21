| type | family | frame | place | n | strict | nometa | partial | acc | base acc (same items) | no-instruction pass | med tokens | empty | trunc | no answer | unjudged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| word_suppression | suppression | F0 | user | 36 | 29/36 | 25/36 | 0.86 | 21/36 | 21/36 | 0/36 | 624 | 0 | 0 | 0 | 0 |
| multiple_word_suppression | suppression | F0 | user | 36 | 17/36 | 15/36 | 0.59 | 17/36 | 21/36 | 0/36 | 399 | 0 | 0 | 0 | 0 |
| repeat_sentences | addition | F0 | user | 36 | 23/36 | 23/36 | 0.82 | 22/36 | 21/36 | 0/36 | 936 | 0 | 0 | 0 | 0 |
| end_of_sentence | addition | F0 | user | 36 | 6/36 | 6/36 | 0.52 | 19/36 | 21/36 | 0/36 | 470 | 0 | 1 | 1 | 0 |
| meow_between_words | addition | F0 | user | 36 | 17/36 | 17/36 | 0.99 | 19/36 | 21/36 | 0/36 | 1124 | 0 | 0 | 2 | 0 |
| lowercase_thinking | stylistic | F0 | user | 36 | 32/36 | 30/36 | 1.00 | 24/36 | 21/36 | 1/36 | 662 | 0 | 0 | 0 | 0 |
| uppercase_thinking | stylistic | F0 | user | 36 | 22/36 | 21/36 | 0.99 | 20/36 | 21/36 | 0/36 | 978 | 0 | 0 | 0 | 0 |
| alternating_case | stylistic | F0 | user | 36 | 11/36 | 11/36 | 0.95 | 21/36 | 21/36 | 0/36 | 963 | 0 | 2 | 2 | 0 |
| ignore_question | ignore | F0 | user | 36 | 4/36 | 3/36 | - | 18/36 | 21/36 | 0/36 | 216 | 1 | 1 | 1 | 0 |
| decoy_topic | ignore | F0 | user | 36 | 30/36 | 30/36 | - | 18/36 | 21/36 | 0/36 | 124 | 0 | 0 | 0 | 0 |
| no_question_restate | ignore | F0 | user | 35 | 24/35 | 24/35 | 0.74 | 19/35 | 20/35 | 19/35 | 875 | 0 | 0 | 0 | 0 |
| no_option_text | ignore | F0 | user | 29 | 19/29 | 19/29 | 0.87 | 18/29 | 15/29 | 13/29 | 673 | 0 | 0 | 0 | 0 |
| no_commitment | commitment | F0 | user | 36 | 2/36 | 2/36 | - | 21/36 | 21/36 | 2/36 | 624 | 0 | 0 | 0 | 0 |
| answer_hidden | commitment | F0 | user | 36 | 2/36 | 2/36 | - | 21/36 | 21/36 | 3/36 | 975 | 0 | 0 | 1 | 0 |
| misdirect | decoupled | F0 | user | 36 | 3/36 | 3/36 | - | 12/36 | 21/36 | 0/36 | 1333 | 0 | 0 | 0 | 0 |
| elimination_only | mode | F0 | user | 36 | 0/36 | 0/36 | - | 20/36 | 21/36 | 0/36 | 614 | 0 | 0 | 0 | 0 |
| no_backtracking | mode | F0 | user | 36 | 12/36 | 12/36 | - | 21/36 | 21/36 | 5/36 | 384 | 0 | 0 | 0 | 0 |
| first_principles_only | mode | F0 | user | 36 | 5/36 | 5/36 | - | 24/36 | 21/36 | 2/36 | 568 | 0 | 0 | 0 | 0 |
| questions_only | mode | F0 | user | 36 | 0/36 | 0/36 | 0.25 | 18/36 | 21/36 | 0/36 | 544 | 0 | 0 | 0 | 0 |
| no_calculation | mode | F0 | user | 36 | 18/36 | 18/36 | 0.57 | 21/36 | 21/36 | 13/36 | 569 | 0 | 0 | 0 | 0 |
| max_words | length | F0 | user | 36 | 22/36 | 22/36 | 0.78 | 21/36 | 21/36 | 3/36 | 94 | 0 | 0 | 0 | 0 |
| no_letter_e | textual | F0 | user | 36 | 1/36 | 1/36 | 0.84 | 21/36 | 21/36 | 0/36 | 421 | 0 | 0 | 0 | 0 |
| third_person | textual | F0 | user | 36 | 12/36 | 11/36 | 0.95 | 22/36 | 21/36 | 2/36 | 470 | 0 | 0 | 0 | 0 |
| end_apple | textual | F0 | user | 36 | 9/36 | 9/36 | 0.67 | 22/36 | 21/36 | 0/36 | 425 | 0 | 0 | 0 | 0 |

baseline arm (no requirement): accuracy 21/36, median tokens 1115

892 generations, model Qwen/Qwen3.8-27B, path tinker://f17512b9-bc5f-588c-b3c6-d4940fcebdd9:train:0/sampler_weights/cotctl-opsd-grpo1b-step10, sampling cost ~$8.23 (price table, excludes judges)
