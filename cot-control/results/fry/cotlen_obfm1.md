| run | no-requirement median tokens (vs gptoss-base) | median over types of (tokens with requirement / without), paired | types with ratio below 0.8 | types above 1.25 | types |
|---|---|---|---|---|---|
| gptoss-base | 594 (1.00x) | 0.92 | 5/24 | 1/24 | 24 |
| mt1b-s100 | 726 (1.22x) | 0.42 | 23/27 | 2/27 | 27 |
| obfm1c-s40 | 612 (1.03x) | 0.54 | 19/27 | 3/27 | 27 |
| mt1b-s150 | 666 (1.12x) | 0.45 | 22/24 | 0/24 | 24 |

Per type, paired median ratio (tokens with the requirement / tokens without, same items):

| type | gptoss-base | mt1b-s100 | obfm1c-s40 | mt1b-s150 |
|---|---|---|---|---|
| alternating_case | 0.95 | 1.20 | 1.52 | 1.15 |
| answer_hidden | 1.11 | 0.78 | 1.00 | 0.64 |
| decoy_topic | 0.95 | 0.65 | 0.68 | 0.75 |
| elimination_only | 1.12 | 0.42 | 0.59 | 0.35 |
| end_apple | 0.85 | 0.36 | 0.44 | 0.37 |
| end_of_sentence | 0.94 | 0.37 | 0.45 | 0.35 |
| first_principles_only | 0.92 | 0.34 | 0.54 | 0.46 |
| ignore_question | 0.65 | 0.29 | 0.29 | 0.37 |
| lowercase_thinking | 0.81 | 0.47 | 0.62 | 0.50 |
| max_words | 0.36 | 0.10 | 0.09 | 0.12 |
| meow_between_words | 0.96 | 1.03 | 1.23 | 0.84 |
| misdirect | 1.23 | 0.62 | 0.83 | 0.63 |
| multiple_word_suppression | 0.90 | 0.27 | 0.37 | 0.38 |
| no_backtracking | 0.81 | 0.41 | 0.51 | 0.47 |
| no_calculation | 0.80 | 0.32 | 0.39 | 0.38 |
| no_commitment | 0.95 | 0.35 | 0.50 | 0.45 |
| no_letter_e | 1.25 | 0.26 | 0.33 | 0.31 |
| no_option_text | 0.91 | 0.32 | 0.45 | 0.33 |
| no_question_restate | 0.78 | 0.27 | 0.34 | 0.24 |
| questions_only | 0.59 | 0.48 | 0.51 | 0.45 |
| repeat_sentences | 0.92 | 0.68 | 0.85 | 0.77 |
| third_person | 0.93 | 0.51 | 0.74 | 0.55 |
| uppercase_thinking | 0.92 | 0.53 | 0.88 | 0.65 |
| word_suppression | 0.86 | 0.33 | 0.48 | 0.42 |
