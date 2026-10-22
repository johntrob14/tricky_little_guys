| type | family | frame | place | n | strict | nometa | partial | acc | base acc (same items) | no-instruction pass | med tokens | empty | trunc | no answer | unjudged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| word_suppression | suppression | F0 | user | 108 | 106/108 | 106/108 | 0.99 | 51/108 | 62/108 | 0/108 | 260 | 0 | 0 | 0 | 0 |
| multiple_word_suppression | suppression | F0 | user | 108 | 88/108 | 88/108 | 0.89 | 46/108 | 62/108 | 0/108 | 272 | 0 | 0 | 0 | 0 |
| repeat_sentences | addition | F0 | user | 108 | 108/108 | 108/108 | 1.00 | 54/108 | 62/108 | 0/108 | 470 | 0 | 0 | 0 | 0 |
| end_of_sentence | addition | F0 | user | 108 | 101/108 | 101/108 | 0.97 | 47/108 | 62/108 | 0/108 | 208 | 0 | 0 | 0 | 0 |
| meow_between_words | addition | F0 | user | 108 | 57/108 | 57/108 | 0.98 | 56/108 | 62/108 | 0/108 | 666 | 0 | 0 | 0 | 0 |
| lowercase_thinking | stylistic | F0 | user | 108 | 86/108 | 86/108 | 1.00 | 57/108 | 62/108 | 0/108 | 356 | 0 | 0 | 0 | 0 |
| uppercase_thinking | stylistic | F0 | user | 108 | 66/108 | 66/108 | 0.99 | 52/108 | 62/108 | 0/108 | 430 | 0 | 0 | 0 | 0 |
| alternating_case | stylistic | F0 | user | 108 | 57/108 | 57/108 | 0.99 | 49/108 | 62/108 | 0/108 | 846 | 0 | 0 | 0 | 0 |
| ignore_question | ignore | F0 | user | 108 | 105/108 | 105/108 | - | 31/108 | 62/108 | 0/108 | 306 | 0 | 0 | 7 | 0 |
| decoy_topic | ignore | F0 | user | 108 | 107/108 | 107/108 | - | 31/108 | 62/108 | 0/108 | 500 | 0 | 0 | 0 | 0 |
| no_question_restate | ignore | F0 | user | 105 | 105/105 | 105/105 | 1.00 | 40/105 | 59/105 | 49/105 | 181 | 0 | 0 | 0 | 0 |
| no_option_text | ignore | F0 | user | 92 | 92/92 | 92/92 | 1.00 | 30/92 | 49/92 | 34/92 | 218 | 0 | 0 | 0 | 0 |
| no_commitment | commitment | F0 | user | 108 | 70/108 | 70/108 | - | 38/108 | 62/108 | 6/108 | 327 | 0 | 0 | 0 | 0 |
| answer_hidden | commitment | F0 | user | 108 | 47/108 | 47/108 | - | 37/108 | 62/108 | 4/108 | 481 | 0 | 0 | 1 | 0 |
| misdirect | decoupled | F0 | user | 108 | 25/108 | 24/108 | - | 40/108 | 62/108 | 0/108 | 454 | 0 | 0 | 0 | 0 |
| elimination_only | mode | F0 | user | 108 | 102/108 | 102/108 | - | 37/108 | 62/108 | 0/108 | 258 | 0 | 0 | 0 | 0 |
| no_backtracking | mode | F0 | user | 108 | 97/108 | 97/108 | - | 57/108 | 62/108 | 52/108 | 276 | 0 | 0 | 0 | 0 |
| first_principles_only | mode | F0 | user | 108 | 45/108 | 45/108 | - | 32/108 | 62/108 | 1/108 | 336 | 0 | 0 | 0 | 0 |
| questions_only | mode | F0 | user | 108 | 105/108 | 103/108 | 1.00 | 57/108 | 62/108 | 0/108 | 316 | 0 | 0 | 0 | 0 |
| no_calculation | mode | F0 | user | 108 | 108/108 | 108/108 | 1.00 | 33/108 | 62/108 | 45/108 | 259 | 0 | 0 | 0 | 0 |
| max_words | length | F0 | user | 108 | 104/108 | 104/108 | 0.99 | 46/108 | 62/108 | 4/108 | 83 | 0 | 0 | 0 | 0 |
| no_letter_e | textual | F0 | user | 108 | 0/108 | 0/108 | 0.73 | 47/108 | 62/108 | 0/108 | 202 | 0 | 0 | 1 | 0 |
| third_person | textual | F0 | user | 108 | 103/108 | 103/108 | 0.99 | 53/108 | 62/108 | 27/108 | 381 | 0 | 0 | 0 | 0 |
| end_apple | textual | F0 | user | 108 | 105/108 | 103/108 | 0.99 | 52/108 | 62/108 | 0/108 | 233 | 0 | 0 | 1 | 0 |

baseline arm (no requirement): accuracy 62/108, median tokens 666

2681 generations, model openai/gpt-oss-120b, path tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step150, sampling cost ~$1.39 (price table, excludes judges)
