| run | no-requirement median tokens (vs gptoss-base) | median over types of (tokens with requirement / without), paired | types with ratio below 0.8 | types above 1.25 | types |
|---|---|---|---|---|---|
| gptoss-base | 594 (1.00x) | 0.92 | 5/24 | 1/24 | 24 |
| gptoss-grpo3s100 | 469 (0.79x) | 0.87 | 11/24 | 2/24 | 24 |
| mt1b-s150 | 666 (1.12x) | 0.45 | 22/24 | 0/24 | 24 |
| gptoss-len2s60 | 322 (0.54x) | 0.75 | 15/25 | 0/25 | 25 |
| gptoss-st1-ignore2 | 198 (0.33x) | 0.63 | 18/27 | 3/27 | 27 |
| gptoss-st1-words | 376 (0.63x) | 0.80 | 13/27 | 2/27 | 27 |
| gptoss-st1-third | 765 (1.29x) | 0.77 | 15/27 | 2/27 | 27 |
| gptoss-st1-qonly | 796 (1.34x) | 0.91 | 6/27 | 3/27 | 27 |

Per type, paired median ratio (tokens with the requirement / tokens without, same items):

| type | gptoss-base | gptoss-grpo3s100 | mt1b-s150 | gptoss-len2s60 | gptoss-st1-ignore2 | gptoss-st1-words | gptoss-st1-third | gptoss-st1-qonly |
|---|---|---|---|---|---|---|---|---|
| alternating_case | 0.95 | 1.20 | 1.15 | 0.74 | 0.33 | 0.98 | 0.88 | 0.93 |
| answer_hidden | 1.11 | 1.08 | 0.64 | 0.92 | 0.97 | 1.05 | 0.97 | 1.13 |
| decoy_topic | 0.95 | 0.57 | 0.75 | 0.60 | 0.83 | 0.62 | 0.47 | 0.31 |
| elimination_only | 1.12 | 1.14 | 0.35 | 1.11 | 0.85 | 1.15 | 1.00 | 1.14 |
| end_apple | 0.85 | 0.69 | 0.37 | 0.66 | 0.65 | 0.66 | 0.70 | 0.88 |
| end_of_sentence | 0.94 | 0.76 | 0.35 | 0.69 | 0.53 | 0.65 | 0.68 | 0.91 |
| first_principles_only | 0.92 | 1.01 | 0.46 | 0.84 | 0.65 | 0.89 | 0.73 | 0.80 |
| ignore_question | 0.65 | 0.59 | 0.37 | 0.55 | 0.16 | 0.39 | 0.37 | 0.18 |
| lowercase_thinking | 0.81 | 0.78 | 0.50 | 0.66 | 0.59 | 0.73 | 0.72 | 0.81 |
| max_words | 0.36 | 0.23 | 0.12 | 0.27 | 0.35 | 0.20 | 0.27 | 0.19 |
| meow_between_words | 0.96 | 1.65 | 0.84 | 0.67 | 1.04 | 0.84 | 0.74 | 1.03 |
| misdirect | 1.23 | 1.33 | 0.63 | 1.16 | 0.50 | 1.07 | 1.01 | 1.35 |
| multiple_word_suppression | 0.90 | 0.79 | 0.38 | 0.75 | 0.39 | 0.68 | 0.71 | 0.83 |
| no_backtracking | 0.81 | 0.71 | 0.47 | 0.74 | 0.63 | 0.73 | 0.73 | 0.79 |
| no_calculation | 0.80 | 0.71 | 0.38 | 0.71 | 0.38 | 0.76 | 0.69 | 0.79 |
| no_commitment | 0.95 | 0.92 | 0.45 | 0.85 | 0.32 | 0.89 | 0.89 | 0.99 |
| no_letter_e | 1.25 | 0.70 | 0.31 | 0.71 | 0.57 | 0.56 | 0.56 | 0.66 |
| no_option_text | 0.91 | 0.85 | 0.33 | 0.79 | 0.73 | 0.80 | 0.84 | 0.91 |
| no_question_restate | 0.78 | 0.76 | 0.24 | 0.78 | 0.68 | 0.77 | 0.74 | 0.84 |
| questions_only | 0.59 | 0.89 | 0.45 | 0.81 | 0.98 | 0.70 | 0.77 | 0.87 |
| repeat_sentences | 0.92 | 1.01 | 0.77 | 0.92 | 1.17 | 0.83 | 0.85 | 0.96 |
| third_person | 0.93 | 0.96 | 0.55 | 0.87 | 0.62 | 0.90 | 0.83 | 0.99 |
| uppercase_thinking | 0.92 | 1.08 | 0.65 | 0.73 | 0.33 | 0.83 | 0.88 | 0.97 |
| word_suppression | 0.86 | 0.98 | 0.42 | 0.82 | 0.59 | 0.74 | 0.77 | 0.87 |
