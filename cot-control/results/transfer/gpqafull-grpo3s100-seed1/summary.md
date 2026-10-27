| type | family | frame | place | n | strict | nometa | partial | acc | base acc (same items) | no-instruction pass | med tokens | empty | trunc | no answer | unjudged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| word_suppression | suppression | F0 | user | 360 | 337/360 | 337/360 | 0.96 | 207/360 | 242/360 | - | 588 | 0 | 0 | 0 | 0 |
| multiple_word_suppression | suppression | F0 | user | 360 | 230/360 | 223/360 | 0.74 | 205/360 | 242/360 | - | 466 | 0 | 0 | 0 | 0 |
| repeat_sentences | addition | F0 | user | 360 | 356/360 | 352/360 | 0.99 | 241/360 | 242/360 | - | 622 | 0 | 0 | 0 | 0 |
| end_of_sentence | addition | F0 | user | 360 | 110/360 | 110/360 | 0.65 | 240/360 | 242/360 | - | 412 | 0 | 0 | 0 | 0 |
| meow_between_words | addition | F0 | user | 360 | 195/360 | 194/360 | 0.97 | 190/360 | 242/360 | - | 932 | 0 | 0 | 0 | 0 |
| lowercase_thinking | stylistic | F0 | user | 360 | 294/360 | 294/360 | 1.00 | 240/360 | 242/360 | - | 518 | 0 | 0 | 0 | 0 |
| uppercase_thinking | stylistic | F0 | user | 360 | 206/360 | 205/360 | 0.99 | 239/360 | 242/360 | - | 560 | 0 | 0 | 0 | 0 |
| alternating_case | stylistic | F0 | user | 360 | 59/360 | 59/360 | 0.97 | 199/360 | 242/360 | - | 662 | 0 | 0 | 0 | 0 |
| ignore_question | ignore | F0 | user | 360 | 258/360 | 249/360 | - | 176/360 | 242/360 | - | 278 | 0 | 0 | 0 | 0 |
| decoy_topic | ignore | F0 | user | 360 | 288/360 | 271/360 | - | 160/360 | 242/360 | - | 291 | 0 | 0 | 0 | 0 |
| no_question_restate | ignore | F0 | user | 360 | 311/360 | 310/360 | 0.90 | 238/360 | 242/360 | - | 502 | 0 | 0 | 0 | 0 |
| no_option_text | ignore | F0 | user | 263 | 226/263 | 223/263 | 0.94 | 176/263 | 176/263 | - | 539 | 0 | 0 | 0 | 0 |
| no_commitment | commitment | F0 | user | 360 | 0/360 | 0/360 | - | 233/360 | 242/360 | - | 605 | 0 | 0 | 0 | 0 |
| answer_hidden | commitment | F0 | user | 360 | 3/360 | 3/360 | - | 237/360 | 242/360 | - | 680 | 0 | 0 | 0 | 0 |
| misdirect | decoupled | F0 | user | 360 | 1/360 | 1/360 | - | 205/360 | 242/360 | - | 822 | 0 | 0 | 0 | 0 |
| no_backtracking | mode | F0 | user | 360 | 70/360 | 70/360 | - | 236/360 | 242/360 | - | 448 | 0 | 0 | 0 | 0 |
| questions_only | mode | F0 | user | 360 | 308/360 | 307/360 | 0.95 | 221/360 | 242/360 | - | 490 | 0 | 0 | 0 | 0 |
| no_calculation | mode | F0 | user | 360 | 162/360 | 162/360 | 0.57 | 217/360 | 242/360 | - | 391 | 0 | 0 | 0 | 0 |
| no_letter_e | textual | F0 | user | 360 | 0/360 | 0/360 | 0.79 | 214/360 | 242/360 | - | 432 | 0 | 0 | 0 | 0 |
| third_person | textual | F0 | user | 360 | 255/360 | 255/360 | 0.98 | 243/360 | 242/360 | - | 594 | 0 | 0 | 0 | 0 |
| end_apple | textual | F0 | user | 360 | 184/360 | 184/360 | 0.83 | 227/360 | 242/360 | - | 392 | 0 | 0 | 0 | 0 |

baseline arm (no requirement): accuracy 242/360, median tokens 706

7823 generations, model openai/gpt-oss-120b, path tinker://ec5bf27e-1436-5923-a854-96774a207315:train:0/sampler_weights/cotctl-grpo3-step100, sampling cost ~$5.84 (price table, excludes judges)
