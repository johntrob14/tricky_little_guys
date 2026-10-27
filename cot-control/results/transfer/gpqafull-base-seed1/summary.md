| type | family | frame | place | n | strict | nometa | partial | acc | base acc (same items) | no-instruction pass | med tokens | empty | trunc | no answer | unjudged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| word_suppression | suppression | F0 | user | 360 | 49/360 | 49/360 | 0.32 | 239/360 | 249/360 | - | 730 | 0 | 0 | 0 | 0 |
| multiple_word_suppression | suppression | F0 | user | 360 | 14/360 | 13/360 | 0.11 | 240/360 | 249/360 | - | 750 | 0 | 0 | 0 | 0 |
| repeat_sentences | addition | F0 | user | 360 | 11/360 | 11/360 | 0.52 | 240/360 | 249/360 | - | 682 | 0 | 0 | 0 | 0 |
| end_of_sentence | addition | F0 | user | 360 | 0/360 | 0/360 | 0.01 | 241/360 | 249/360 | - | 712 | 0 | 0 | 0 | 0 |
| meow_between_words | addition | F0 | user | 360 | 0/360 | 0/360 | 0.02 | 247/360 | 249/360 | - | 678 | 0 | 0 | 0 | 0 |
| lowercase_thinking | stylistic | F0 | user | 360 | 0/360 | 0/360 | 0.93 | 237/360 | 249/360 | - | 628 | 0 | 0 | 0 | 0 |
| uppercase_thinking | stylistic | F0 | user | 360 | 1/360 | 1/360 | 0.09 | 247/360 | 249/360 | - | 697 | 0 | 0 | 0 | 0 |
| alternating_case | stylistic | F0 | user | 360 | 0/360 | 0/360 | 0.25 | 230/360 | 249/360 | - | 1058 | 0 | 0 | 0 | 0 |
| ignore_question | ignore | F0 | user | 360 | 96/360 | 14/360 | - | 214/360 | 249/360 | - | 260 | 0 | 0 | 0 | 1 |
| decoy_topic | ignore | F0 | user | 360 | 2/360 | 1/360 | - | 227/360 | 249/360 | - | 508 | 0 | 0 | 1 | 0 |
| no_question_restate | ignore | F0 | user | 360 | 224/360 | 224/360 | 0.69 | 241/360 | 249/360 | - | 737 | 0 | 0 | 0 | 0 |
| no_option_text | ignore | F0 | user | 263 | 142/263 | 140/263 | 0.70 | 180/263 | 184/263 | - | 669 | 0 | 0 | 0 | 0 |
| no_commitment | commitment | F0 | user | 360 | 4/360 | 4/360 | - | 247/360 | 249/360 | - | 770 | 0 | 0 | 0 | 0 |
| answer_hidden | commitment | F0 | user | 360 | 8/360 | 7/360 | - | 240/360 | 249/360 | - | 774 | 0 | 0 | 0 | 0 |
| misdirect | decoupled | F0 | user | 360 | 5/360 | 3/360 | - | 205/360 | 249/360 | - | 792 | 0 | 0 | 0 | 0 |
| no_backtracking | mode | F0 | user | 360 | 64/360 | 64/360 | - | 247/360 | 249/360 | - | 622 | 0 | 0 | 0 | 0 |
| questions_only | mode | F0 | user | 360 | 58/360 | 58/360 | 0.52 | 211/360 | 249/360 | - | 354 | 0 | 0 | 1 | 0 |
| no_calculation | mode | F0 | user | 360 | 77/360 | 77/360 | 0.30 | 242/360 | 249/360 | - | 598 | 0 | 0 | 0 | 0 |
| no_letter_e | textual | F0 | user | 360 | 0/360 | 0/360 | 0.61 | 238/360 | 249/360 | - | 1142 | 0 | 0 | 0 | 0 |
| third_person | textual | F0 | user | 360 | 36/360 | 36/360 | 0.90 | 242/360 | 249/360 | - | 705 | 0 | 0 | 0 | 0 |
| end_apple | textual | F0 | user | 360 | 0/360 | 0/360 | 0.03 | 240/360 | 249/360 | - | 716 | 0 | 0 | 0 | 0 |

baseline arm (no requirement): accuracy 249/360, median tokens 858

7823 generations, model openai/gpt-oss-120b, path None, sampling cost ~$8.51 (price table, excludes judges)
