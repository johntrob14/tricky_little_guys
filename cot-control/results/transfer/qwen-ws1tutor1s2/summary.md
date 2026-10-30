| type | family | frame | place | n | strict | nometa | partial | acc | base acc (same items) | no-instruction pass | med tokens | empty | trunc | no answer | unjudged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| word_suppression | suppression | F0 | user | 36 | 0/36 | 0/36 | 0.07 | 24/36 | 23/36 | - | 6302 | 0 | 5 | 5 | 0 |
| multiple_word_suppression | suppression | F0 | user | 36 | 0/36 | 0/36 | 0.05 | 24/36 | 23/36 | - | 3964 | 0 | 6 | 6 | 0 |
| repeat_sentences | addition | F0 | user | 36 | 0/36 | 0/36 | 0.01 | 22/36 | 23/36 | - | 1930 | 4 | 5 | 7 | 0 |
| end_of_sentence | addition | F0 | user | 36 | 0/36 | 0/36 | 0.09 | 23/36 | 23/36 | - | 2970 | 0 | 5 | 5 | 0 |
| meow_between_words | addition | F0 | user | 36 | 0/36 | 0/36 | 0.04 | 23/36 | 23/36 | - | 2414 | 0 | 7 | 7 | 0 |
| lowercase_thinking | stylistic | F0 | user | 36 | 0/36 | 0/36 | 0.92 | 24/36 | 23/36 | - | 2844 | 0 | 8 | 8 | 0 |
| uppercase_thinking | stylistic | F0 | user | 36 | 2/36 | 1/36 | 0.13 | 25/36 | 23/36 | - | 1836 | 0 | 5 | 5 | 0 |
| alternating_case | stylistic | F0 | user | 36 | 0/36 | 0/36 | 0.20 | 23/36 | 23/36 | - | 7010 | 0 | 7 | 7 | 0 |
| ignore_question | ignore | F0 | user | 36 | 0/36 | 0/36 | - | 24/36 | 23/36 | - | 452 | 0 | 3 | 3 | 0 |
| decoy_topic | ignore | F0 | user | 36 | 1/36 | 0/36 | - | 22/36 | 23/36 | - | 1150 | 0 | 4 | 4 | 0 |
| no_question_restate | ignore | F0 | user | 35 | 10/35 | 4/35 | 0.33 | 25/35 | 22/35 | - | 3604 | 0 | 3 | 3 | 0 |
| no_option_text | ignore | F0 | user | 29 | 10/29 | 1/29 | 0.46 | 18/29 | 18/29 | - | 4233 | 0 | 5 | 5 | 0 |
| no_commitment | commitment | F0 | user | 36 | 1/36 | 1/36 | - | 24/36 | 23/36 | - | 3584 | 0 | 5 | 5 | 0 |
| answer_hidden | commitment | F0 | user | 36 | 0/36 | 0/36 | - | 24/36 | 23/36 | - | 5332 | 0 | 8 | 8 | 0 |
| misdirect | decoupled | F0 | user | 36 | 1/36 | 0/36 | - | 23/36 | 23/36 | - | 7091 | 0 | 6 | 6 | 0 |
| elimination_only | mode | F0 | user | 36 | 1/36 | 0/36 | - | 24/36 | 23/36 | - | 4673 | 0 | 5 | 5 | 0 |
| no_backtracking | mode | F0 | user | 36 | 2/36 | 2/36 | - | 25/36 | 23/36 | - | 2950 | 0 | 4 | 4 | 0 |
| first_principles_only | mode | F0 | user | 36 | 1/36 | 1/36 | - | 25/36 | 23/36 | - | 2614 | 0 | 6 | 6 | 0 |
| questions_only | mode | F0 | user | 36 | 0/36 | 0/36 | 0.26 | 22/36 | 23/36 | - | 3432 | 0 | 7 | 7 | 0 |
| no_calculation | mode | F0 | user | 36 | 8/36 | 3/36 | 0.29 | 24/36 | 23/36 | - | 2804 | 0 | 5 | 5 | 0 |
| max_words | length | F0 | user | 36 | 11/36 | 2/36 | 0.43 | 25/36 | 23/36 | - | 578 | 0 | 2 | 2 | 0 |
| target_words | length | F0 | user | 36 | 0/36 | 0/36 | 0.00 | 24/36 | 23/36 | - | 5132 | 0 | 4 | 4 | 0 |
| target_tokens | length | F0 | user | 36 | 3/36 | 0/36 | 0.15 | 27/36 | 23/36 | - | 4080 | 0 | 3 | 3 | 0 |
| think_tokens | length | F0 | user | 36 | 1/36 | 0/36 | 0.18 | 26/36 | 23/36 | - | 3565 | 0 | 4 | 4 | 0 |
| no_letter_e | textual | F0 | user | 36 | 0/36 | 0/36 | 0.59 | 26/36 | 23/36 | - | 4251 | 0 | 5 | 5 | 0 |
| third_person | textual | F0 | user | 36 | 1/36 | 0/36 | 0.86 | 27/36 | 23/36 | - | 2943 | 0 | 5 | 5 | 0 |
| end_apple | textual | F0 | user | 36 | 0/36 | 0/36 | 0.11 | 24/36 | 23/36 | - | 2816 | 0 | 6 | 6 | 0 |

baseline arm (no requirement): accuracy 23/36, median tokens 5242

1000 generations, model Qwen/Qwen3.8-27B, path tinker://8f2c68f3-be86-5169-9b22-d3718836e01f:train:0/sampler_weights/arena-turn1-ws1tutor1-step2, sampling cost ~$45.60 (price table, excludes judges)
