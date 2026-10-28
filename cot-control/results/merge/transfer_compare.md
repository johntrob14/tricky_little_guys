Strict passes (meta-discussion allowed), with mean partial compliance in brackets where the type has a graded score. 'trained by' lists the components whose RL trained that type.

| type | trained by | base (Tinker) | base (vLLM) | merge6 (vLLM) | grpo3 s100 | manytypes1b s150 | manytypes2e s170 | len2 s60 |
|---|---|---|---|---|---|---|---|---|
| word_suppression | grpo3, manytypes1b, manytypes2e | 9/108 (0.26) | 14/108 (0.29) | 76/108 (0.80) | 102/108 (0.97) | 106/108 (0.99) | 91/108 (0.90) | 39/108 (0.57) |
| multiple_word_suppression | grpo3, manytypes1b, manytypes2e | 5/108 (0.12) | 4/108 (0.11) | 30/108 (0.41) | 67/108 (0.73) | 88/108 (0.89) | 68/108 (0.73) | 10/108 (0.22) |
| repeat_sentences | grpo3, manytypes1b, manytypes2e | 10/108 (0.54) | 6/108 (0.52) | 91/108 (0.92) | 103/108 (0.98) | 108/108 (1.00) | 72/108 (0.83) | 36/108 (0.67) |
| end_of_sentence | - | 0/108 (0.02) | 0/108 (0.03) | 35/108 (0.51) | 55/108 (0.75) | 101/108 (0.97) | 53/108 (0.78) | 2/108 (0.14) |
| meow_between_words | grpo3, manytypes1b, manytypes2e | 0/108 (0.04) | 0/108 (0.04) | 14/108 (0.85) | 61/108 (0.99) | 57/108 (0.98) | 42/108 (0.95) | 1/108 (0.14) |
| lowercase_thinking | grpo3, manytypes1b, manytypes2e | 0/108 (0.94) | 0/108 (0.93) | 36/108 (0.98) | 97/108 (1.00) | 86/108 (1.00) | 81/108 (1.00) | 0/108 (0.94) |
| uppercase_thinking | - | 2/108 (0.11) | 4/108 (0.12) | 37/108 (0.78) | 69/108 (0.99) | 66/108 (0.99) | 46/108 (0.95) | 6/108 (0.28) |
| alternating_case | grpo3, manytypes1b, manytypes2e | 0/108 (0.23) | 0/108 (0.24) | 16/108 (0.88) | 31/108 (0.98) | 57/108 (0.99) | 19/108 (0.91) | 1/108 (0.42) |
| ignore_question | - | 17/108 | 23/108 | 83/108 | 81/108 | 105/108 | 106/108 | 82/108 |
| decoy_topic | manytypes2e | 0/108 | 0/108 | 98/108 | 89/108 | 107/108 | 105/108 | 11/108 |
| no_question_restate | - | 68/105 (0.71) | 66/105 (0.70) | 84/105 (0.84) | 93/105 (0.91) | 105/105 (1.00) | 85/105 (0.84) | 87/105 (0.87) |
| no_option_text | - | 43/92 (0.71) | 46/92 (0.75) | 57/92 (0.86) | 79/92 (0.96) | 92/92 (1.00) | 72/92 (0.91) | 61/92 (0.90) |
| no_commitment | manytypes1b, manytypes2e | 3/108 | 2/108 | 3/108 | 0/108 | 70/108 | 5/108 | 0/108 |
| answer_hidden | - | 3/108 | 1/108 | 3/108 | 1/108 | 47/108 | 1/108 | 0/108 |
| misdirect | - | 1/108 | 2/108 | 0/108 | 1/108 | 25/108 | 0/108 | 1/108 |
| elimination_only | manytypes1b | 0/108 | 0/108 | 0/108 | 0/108 | 102/108 | - | 0/108 |
| no_backtracking | manytypes1b, manytypes2e | 25/108 | 26/108 | 36/108 | 26/108 | 97/108 | 33/108 | 31/108 |
| first_principles_only | - | 6/108 | 5/108 | 9/108 | 10/108 | 45/108 | - | 6/108 |
| questions_only | manytypes1b, manytypes2e | 20/108 (0.55) | 30/108 (0.55) | 97/108 (0.97) | 92/108 (0.96) | 105/108 (1.00) | 91/108 (0.92) | 71/108 (0.90) |
| no_calculation | manytypes1b | 44/108 (0.50) | 35/108 (0.43) | 64/108 (0.67) | 76/108 (0.77) | 108/108 (1.00) | 83/108 (0.82) | 54/108 (0.61) |
| max_words | manytypes1b | 41/108 (0.64) | 42/108 (0.64) | 83/108 (0.91) | 70/108 (0.86) | 104/108 (0.99) | - | 96/108 (0.96) |
| target_words | - | - | 24/108 (0.40) | 32/108 (0.48) | - | - | - | 37/108 (0.63) |
| target_tokens | - | - | 21/108 (0.36) | 24/108 (0.32) | - | - | - | - |
| think_tokens | - | - | 17/108 (0.33) | 19/108 (0.35) | - | - | - | - |
| no_letter_e | - | 0/108 (0.61) | 0/108 (0.59) | 0/108 (0.70) | 0/108 (0.83) | 0/108 (0.73) | 0/108 (0.87) | 0/108 (0.60) |
| third_person | - | 11/108 (0.89) | 13/108 (0.88) | 50/108 (0.95) | 79/108 (0.98) | 103/108 (0.99) | 60/108 (0.96) | 40/108 (0.94) |
| end_apple | - | 0/108 (0.04) | 0/108 (0.03) | 52/108 (0.65) | 66/108 (0.87) | 105/108 (0.99) | 64/108 (0.80) | 5/108 (0.20) |
| think_language | lang2 | - | 3/108 (0.09) | 85/108 (0.96) | - | - | - | - |
| respond_french | - | - | 0/108 (0.00) | 45/108 (0.58) | - | - | - | - |
| respond_french_front | - | - | 0/108 (0.00) | 46/108 (0.61) | - | - | - | - |
| respond_think_french | - | - | 0/108 (0.00) | 75/108 (0.92) | - | - | - | - |
| answer_language | - | - | 0/108 (0.01) | 67/108 (0.64) | - | - | - | - |
| no-requirement accuracy (baseline arm) | | 59/108 | 63/108 | 57/108 | 57/108 | 62/108 | 61/108 | 55/108 |
| sum over 21 types in every run | | 261/2249 | 272/2249 | 962/2249 | 1268/2249 | 1738/2249 | 1177/2249 | 538/2249 |
| sum over 10 of those, trained by no component | | 145/1061 | 155/1061 | 401/1061 | 524/1061 | 749/1061 | 487/1061 | 284/1061 |

Types trained by no component: end_of_sentence, uppercase_thinking, ignore_question, no_question_restate, no_option_text, answer_hidden, misdirect, no_letter_e, third_person, end_apple.
