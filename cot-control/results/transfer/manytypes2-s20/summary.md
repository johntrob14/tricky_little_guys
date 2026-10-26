| type | family | frame | place | n | strict | nometa | partial | acc | base acc (same items) | no-instruction pass | med tokens | empty | trunc | no answer | unjudged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| word_suppression | suppression | F0 | user | 108 | 16/108 | 15/108 | 0.32 | 58/108 | 58/108 | - | 578 | 0 | 0 | 0 | 0 |
| multiple_word_suppression | suppression | F0 | user | 108 | 8/108 | 7/108 | 0.14 | 58/108 | 58/108 | - | 656 | 0 | 0 | 0 | 0 |
| repeat_sentences | addition | F0 | user | 108 | 9/108 | 9/108 | 0.54 | 60/108 | 58/108 | - | 629 | 0 | 0 | 0 | 0 |
| end_of_sentence | addition | F0 | user | 108 | 0/108 | 0/108 | 0.01 | 57/108 | 58/108 | - | 520 | 0 | 0 | 0 | 0 |
| meow_between_words | addition | F0 | user | 108 | 1/108 | 1/108 | 0.08 | 63/108 | 58/108 | - | 484 | 0 | 0 | 0 | 0 |
| lowercase_thinking | stylistic | F0 | user | 108 | 0/108 | 0/108 | 0.94 | 55/108 | 58/108 | - | 479 | 0 | 0 | 0 | 0 |
| uppercase_thinking | stylistic | F0 | user | 108 | 6/108 | 6/108 | 0.16 | 55/108 | 58/108 | - | 496 | 0 | 0 | 0 | 0 |
| alternating_case | stylistic | F0 | user | 108 | 0/108 | 0/108 | 0.33 | 56/108 | 58/108 | - | 567 | 0 | 0 | 0 | 0 |
| ignore_question | ignore | F0 | user | 108 | 55/108 | 23/108 | - | 45/108 | 58/108 | - | 94 | 0 | 0 | 0 | 0 |
| decoy_topic | ignore | F0 | user | 108 | 1/108 | 0/108 | - | 53/108 | 58/108 | - | 372 | 0 | 0 | 0 | 0 |
| no_question_restate | ignore | F0 | user | 105 | 69/105 | 69/105 | 0.73 | 55/105 | 55/105 | - | 573 | 0 | 0 | 0 | 0 |
| no_option_text | ignore | F0 | user | 92 | 46/92 | 45/92 | 0.75 | 46/92 | 45/92 | - | 503 | 0 | 0 | 0 | 0 |
| no_commitment | commitment | F0 | user | 108 | 3/108 | 2/108 | - | 62/108 | 58/108 | - | 708 | 0 | 0 | 0 | 0 |
| answer_hidden | commitment | F0 | user | 108 | 2/108 | 2/108 | - | 58/108 | 58/108 | - | 638 | 0 | 0 | 0 | 0 |
| misdirect | decoupled | F0 | user | 108 | 2/108 | 2/108 | - | 47/108 | 58/108 | - | 694 | 0 | 0 | 0 | 0 |
| no_backtracking | mode | F0 | user | 108 | 20/108 | 20/108 | - | 60/108 | 58/108 | - | 493 | 0 | 0 | 0 | 0 |
| questions_only | mode | F0 | user | 108 | 51/108 | 51/108 | 0.79 | 52/108 | 58/108 | - | 244 | 0 | 0 | 0 | 0 |
| no_calculation | mode | F0 | user | 108 | 38/108 | 38/108 | 0.45 | 57/108 | 58/108 | - | 500 | 0 | 0 | 0 | 0 |
| no_letter_e | textual | F0 | user | 108 | 0/108 | 0/108 | 0.59 | 58/108 | 58/108 | - | 622 | 0 | 0 | 0 | 0 |
| third_person | textual | F0 | user | 108 | 21/108 | 21/108 | 0.90 | 63/108 | 58/108 | - | 618 | 0 | 0 | 0 | 0 |
| end_apple | textual | F0 | user | 108 | 0/108 | 0/108 | 0.02 | 59/108 | 58/108 | - | 475 | 0 | 0 | 0 | 0 |

baseline arm (no requirement): accuracy 58/108, median tokens 602

2357 generations, model openai/gpt-oss-120b, path tinker://eb4c6396-a0a5-58c7-9584-42e5e109289d:train:0/sampler_weights/cotctl-manytypes2-step20, sampling cost ~$2.48 (price table, excludes judges)
