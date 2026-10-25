| run | no-requirement median tokens (vs qwen-base) | median over types of (tokens with requirement / without), paired | types with ratio below 0.8 | types above 1.25 | types |
|---|---|---|---|---|---|
| qwen-base | 3183 (1.00x) | 1.00 | 2/24 | 2/24 | 24 |
| qwen-opsdgrpo | 1115 (0.35x) | 0.59 | 20/24 | 1/24 | 24 |

Per type, paired median ratio (tokens with the requirement / tokens without, same items):

| type | qwen-base | qwen-opsdgrpo |
|---|---|---|
| alternating_case | 1.33 | 0.97 |
| answer_hidden | 1.14 | 0.82 |
| decoy_topic | 0.90 | 0.13 |
| elimination_only | 1.00 | 0.60 |
| end_apple | 1.00 | 0.38 |
| end_of_sentence | 1.00 | 0.50 |
| first_principles_only | 1.00 | 0.68 |
| ignore_question | 0.52 | 0.22 |
| lowercase_thinking | 1.00 | 0.69 |
| max_words | 0.38 | 0.13 |
| meow_between_words | 1.05 | 1.29 |
| misdirect | 1.83 | 1.09 |
| multiple_word_suppression | 1.08 | 0.54 |
| no_backtracking | 0.95 | 0.46 |
| no_calculation | 1.02 | 0.44 |
| no_commitment | 1.09 | 0.63 |
| no_letter_e | 1.14 | 0.41 |
| no_option_text | 1.00 | 0.55 |
| no_question_restate | 1.00 | 0.65 |
| questions_only | 1.00 | 0.58 |
| repeat_sentences | 1.00 | 0.77 |
| third_person | 1.00 | 0.51 |
| uppercase_thinking | 1.00 | 0.77 |
| word_suppression | 1.10 | 0.63 |
