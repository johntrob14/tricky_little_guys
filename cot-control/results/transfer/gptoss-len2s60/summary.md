| type | family | frame | place | n | strict | nometa | partial | acc | base acc (same items) | no-instruction pass | med tokens | empty | trunc | no answer | unjudged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| word_suppression | suppression | F0 | user | 108 | 39/108 | 34/108 | 0.57 | 52/108 | 55/108 | - | 256 | 0 | 0 | 0 | 0 |
| multiple_word_suppression | suppression | F0 | user | 108 | 10/108 | 8/108 | 0.22 | 58/108 | 55/108 | - | 210 | 0 | 0 | 0 | 0 |
| repeat_sentences | addition | F0 | user | 108 | 36/108 | 35/108 | 0.67 | 54/108 | 55/108 | - | 264 | 0 | 0 | 0 | 0 |
| end_of_sentence | addition | F0 | user | 108 | 2/108 | 2/108 | 0.14 | 53/108 | 55/108 | - | 196 | 0 | 0 | 0 | 0 |
| meow_between_words | addition | F0 | user | 108 | 1/108 | 1/108 | 0.14 | 60/108 | 55/108 | - | 212 | 0 | 0 | 0 | 0 |
| lowercase_thinking | stylistic | F0 | user | 108 | 0/108 | 0/108 | 0.94 | 56/108 | 55/108 | - | 203 | 0 | 0 | 0 | 0 |
| uppercase_thinking | stylistic | F0 | user | 108 | 6/108 | 6/108 | 0.28 | 59/108 | 55/108 | - | 224 | 0 | 0 | 0 | 0 |
| alternating_case | stylistic | F0 | user | 108 | 1/108 | 1/108 | 0.42 | 54/108 | 55/108 | - | 238 | 0 | 0 | 0 | 0 |
| ignore_question | ignore | F0 | user | 108 | 82/108 | 45/108 | - | 36/108 | 55/108 | - | 170 | 0 | 0 | 0 | 0 |
| decoy_topic | ignore | F0 | user | 108 | 11/108 | 2/108 | - | 41/108 | 55/108 | - | 172 | 0 | 0 | 0 | 0 |
| no_question_restate | ignore | F0 | user | 105 | 87/105 | 87/105 | 0.87 | 51/105 | 52/105 | - | 259 | 0 | 0 | 0 | 0 |
| no_option_text | ignore | F0 | user | 92 | 61/92 | 61/92 | 0.90 | 45/92 | 43/92 | - | 224 | 0 | 0 | 0 | 0 |
| no_commitment | commitment | F0 | user | 108 | 0/108 | 0/108 | - | 60/108 | 55/108 | - | 253 | 0 | 0 | 0 | 0 |
| answer_hidden | commitment | F0 | user | 108 | 0/108 | 0/108 | - | 59/108 | 55/108 | - | 282 | 0 | 0 | 0 | 0 |
| misdirect | decoupled | F0 | user | 108 | 1/108 | 0/108 | - | 25/108 | 55/108 | - | 389 | 0 | 0 | 0 | 0 |
| elimination_only | mode | F0 | user | 108 | 0/108 | 0/108 | - | 54/108 | 55/108 | - | 318 | 0 | 0 | 0 | 0 |
| no_backtracking | mode | F0 | user | 108 | 31/108 | 31/108 | - | 57/108 | 55/108 | - | 218 | 0 | 0 | 0 | 0 |
| first_principles_only | mode | F0 | user | 108 | 6/108 | 6/108 | - | 55/108 | 55/108 | - | 252 | 0 | 0 | 0 | 0 |
| questions_only | mode | F0 | user | 108 | 71/108 | 71/108 | 0.90 | 47/108 | 55/108 | - | 234 | 0 | 0 | 0 | 0 |
| no_calculation | mode | F0 | user | 108 | 54/108 | 54/108 | 0.61 | 52/108 | 55/108 | - | 220 | 0 | 0 | 0 | 0 |
| max_words | length | F0 | user | 108 | 96/108 | 96/108 | 0.96 | 53/108 | 55/108 | - | 68 | 0 | 0 | 0 | 0 |
| target_words | length | F0 | user | 108 | 37/108 | 37/108 | 0.63 | 60/108 | 55/108 | - | 322 | 0 | 0 | 0 | 0 |
| no_letter_e | textual | F0 | user | 108 | 0/108 | 0/108 | 0.60 | 55/108 | 55/108 | - | 238 | 0 | 0 | 0 | 0 |
| third_person | textual | F0 | user | 108 | 40/108 | 40/108 | 0.94 | 60/108 | 55/108 | - | 275 | 0 | 0 | 0 | 0 |
| end_apple | textual | F0 | user | 108 | 5/108 | 5/108 | 0.20 | 53/108 | 55/108 | - | 204 | 0 | 0 | 0 | 0 |

baseline arm (no requirement): accuracy 55/108, median tokens 322

2789 generations, model openai/gpt-oss-120b, path tinker://b7a21354-dcbc-5b80-9bd5-f10f690ee7f1:train:0/sampler_weights/cotlen-len2-step60, sampling cost ~$1.13 (price table, excludes judges)
