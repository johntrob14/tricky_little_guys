| type | family | frame | place | n | strict | nometa | partial | acc | base acc (same items) | no-instruction pass | med tokens | empty | trunc | no answer | unjudged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| word_suppression | suppression | F0 | user | 36 | 21/36 | 19/36 | 0.65 | 26/36 | 27/36 | - | 1098 | 0 | 4 | 4 | 0 |
| meow_between_words | addition | F0 | user | 36 | 20/36 | 17/36 | 0.85 | 25/36 | 27/36 | - | 3680 | 0 | 7 | 7 | 0 |
| alternating_case | stylistic | F0 | user | 36 | 11/36 | 11/36 | 0.99 | 25/36 | 27/36 | - | 3335 | 0 | 7 | 7 | 0 |
| ignore_question | ignore | F0 | user | 36 | 32/36 | 32/36 | - | 17/36 | 27/36 | - | 232 | 0 | 5 | 5 | 0 |
| no_backtracking | mode | F0 | user | 36 | 22/36 | 22/36 | - | 25/36 | 27/36 | - | 297 | 0 | 1 | 1 | 0 |
| max_words | length | F0 | user | 36 | 28/36 | 28/36 | 0.94 | 20/36 | 27/36 | - | 76 | 0 | 0 | 0 | 0 |

baseline arm (no requirement): accuracy 27/36, median tokens 1412

252 generations, model deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144, path tinker://b6ffa062-0433-5852-9f29-00662919570c:train:0/sampler_weights/arena-ds41sft1-ep2, sampling cost ~$4.45 (price table, excludes judges)
