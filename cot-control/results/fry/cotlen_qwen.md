| run | no-requirement median tokens (vs qwen-base) | median over types of (tokens with requirement / without), paired | types with ratio below 0.8 | types above 1.25 | types |
|---|---|---|---|---|---|
| qwen-base | 3183 (1.00x) | 1.00 | 2/24 | 2/24 | 24 |
| qwen-opsd1s30 | 2406 (0.76x) | 0.93 | 10/27 | 3/27 | 27 |
| qwen-opsdgrpo | 1115 (0.35x) | 0.59 | 20/24 | 1/24 | 24 |
| qwen-ws1tutor1s2 | 5242 (1.65x) | 1.00 | 4/27 | 0/27 | 27 |

Per type, paired median ratio (tokens with the requirement / tokens without, same items):

| type | qwen-base | qwen-opsd1s30 | qwen-opsdgrpo | qwen-ws1tutor1s2 |
|---|---|---|---|---|
| alternating_case | 1.33 | 1.63 | 0.97 | 1.00 |
| answer_hidden | 1.14 | 1.15 | 0.82 | 1.00 |
| decoy_topic | 0.90 | 0.21 | 0.13 | 0.53 |
| elimination_only | 1.00 | 1.21 | 0.60 | 1.00 |
| end_apple | 1.00 | 0.63 | 0.38 | 1.00 |
| end_of_sentence | 1.00 | 0.72 | 0.50 | 1.00 |
| first_principles_only | 1.00 | 1.07 | 0.68 | 0.86 |
| ignore_question | 0.52 | 0.29 | 0.22 | 0.41 |
| lowercase_thinking | 1.00 | 0.94 | 0.69 | 1.00 |
| max_words | 0.38 | 0.36 | 0.13 | 0.38 |
| meow_between_words | 1.05 | 1.77 | 1.29 | 1.00 |
| misdirect | 1.83 | 1.59 | 1.09 | 1.07 |
| multiple_word_suppression | 1.08 | 0.93 | 0.54 | 1.00 |
| no_backtracking | 0.95 | 0.74 | 0.46 | 0.84 |
| no_calculation | 1.02 | 0.77 | 0.44 | 0.86 |
| no_commitment | 1.09 | 0.98 | 0.63 | 1.00 |
| no_letter_e | 1.14 | 0.93 | 0.41 | 1.00 |
| no_option_text | 1.00 | 1.19 | 0.55 | 0.98 |
| no_question_restate | 1.00 | 0.87 | 0.65 | 0.83 |
| questions_only | 1.00 | 0.77 | 0.58 | 1.00 |
| repeat_sentences | 1.00 | 1.15 | 0.77 | 0.66 |
| third_person | 1.00 | 1.21 | 0.51 | 1.00 |
| uppercase_thinking | 1.00 | 0.81 | 0.77 | 1.00 |
| word_suppression | 1.10 | 1.15 | 0.63 | 1.06 |
