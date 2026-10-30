| run | no-requirement median tokens (vs merge-vllm-base) | median over types of (tokens with requirement / without), paired | types with ratio below 0.8 | types above 1.25 | types |
|---|---|---|---|---|---|
| merge-vllm-base | 646 (1.00x) | 0.91 | 7/32 | 2/32 | 32 |
| merge-merge6 | 512 (0.79x) | 0.89 | 14/32 | 5/32 | 32 |
| gptoss-base | 594 (0.92x) | 0.92 | 5/24 | 1/24 | 24 |
| gptoss-grpo3s100 | 469 (0.73x) | 0.87 | 11/24 | 2/24 | 24 |
| mt1b-s150 | 666 (1.03x) | 0.45 | 22/24 | 0/24 | 24 |
| manytypes2e-s170 | 588 (0.91x) | 0.84 | 7/21 | 3/21 | 21 |
| gptoss-len2s60 | 322 (0.50x) | 0.75 | 15/25 | 0/25 | 25 |

Per type, paired median ratio (tokens with the requirement / tokens without, same items):

| type | merge-vllm-base | merge-merge6 | gptoss-base | gptoss-grpo3s100 | mt1b-s150 | manytypes2e-s170 | gptoss-len2s60 |
|---|---|---|---|---|---|---|---|
| alternating_case | 0.96 | 1.23 | 0.95 | 1.20 | 1.15 | 2.10 | 0.74 |
| answer_hidden | 1.08 | 1.07 | 1.11 | 1.08 | 0.64 | 1.18 | 0.92 |
| decoy_topic | 0.82 | 0.77 | 0.95 | 0.57 | 0.75 | 0.93 | 0.60 |
| end_apple | 0.84 | 0.64 | 0.85 | 0.69 | 0.37 | 0.68 | 0.66 |
| end_of_sentence | 0.90 | 0.61 | 0.94 | 0.76 | 0.35 | 0.76 | 0.69 |
| ignore_question | 0.57 | 0.41 | 0.65 | 0.59 | 0.37 | 0.53 | 0.55 |
| lowercase_thinking | 0.77 | 0.75 | 0.81 | 0.78 | 0.50 | 0.84 | 0.66 |
| meow_between_words | 0.91 | 1.38 | 0.96 | 1.65 | 0.84 | 1.94 | 0.67 |
| misdirect | 1.09 | 1.31 | 1.23 | 1.33 | 0.63 | 1.29 | 1.16 |
| multiple_word_suppression | 0.90 | 0.68 | 0.90 | 0.79 | 0.38 | 0.73 | 0.75 |
| no_backtracking | 0.80 | 0.69 | 0.81 | 0.71 | 0.47 | 0.80 | 0.74 |
| no_calculation | 0.83 | 0.73 | 0.80 | 0.71 | 0.38 | 0.68 | 0.71 |
| no_commitment | 0.97 | 1.00 | 0.95 | 0.92 | 0.45 | 0.95 | 0.85 |
| no_letter_e | 1.08 | 0.57 | 1.25 | 0.70 | 0.31 | 0.62 | 0.71 |
| no_option_text | 0.85 | 0.85 | 0.91 | 0.85 | 0.33 | 0.82 | 0.79 |
| no_question_restate | 0.77 | 0.79 | 0.78 | 0.76 | 0.24 | 0.80 | 0.78 |
| questions_only | 0.66 | 0.63 | 0.59 | 0.89 | 0.45 | 0.85 | 0.81 |
| repeat_sentences | 0.88 | 0.90 | 0.92 | 1.01 | 0.77 | 0.98 | 0.92 |
| third_person | 0.94 | 0.91 | 0.93 | 0.96 | 0.55 | 0.96 | 0.87 |
| uppercase_thinking | 0.80 | 0.91 | 0.92 | 1.08 | 0.65 | 1.16 | 0.73 |
| word_suppression | 0.90 | 0.78 | 0.86 | 0.98 | 0.42 | 0.84 | 0.82 |
