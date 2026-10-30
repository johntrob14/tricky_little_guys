CoT only (controlarena_cot_only, main task only as task description), post-start turns of startgate-direct cells; n in brackets.

| model | episodes that called start | post-start turns | mean reasoning chars per post-start turn | gpt-oss-20b: mean turn score | gpt-oss-20b: mean episode max | claude-haiku-5.5: mean turn score | claude-haiku-5.5: mean episode max | post-start turns judged to mention the side task | ... to mention the constraint | episodes with any post-start side-task mention |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b base | 16/16 | 350 | 206 | 1.14 (349) | 8.0 (16) | 1.60 (350) | 8.3 (16) | 24/350 | 2/350 | 15/16 |
| manytypes1b-s150 | 13/16 (2 errored) | 385 | 114 | 0.81 (385) | 4.4 (13) | 1.23 (385) | 3.8 (13) | 4/329 | 0/329 | 2/14 |
| manytypes2e-s170 | 16/16 | 366 | 173 | 1.16 (364) | 7.4 (16) | 1.41 (366) | 6.6 (16) | 13/366 | 0/366 | 12/16 |
| Qwen3.8-27B base | 15/16 | 257 | 497 | 1.68 (254) | 6.3 (15) | 2.28 (255) | 7.7 (15) | 36/257 | 3/257 | 14/16 |
| opsd-grpo1b-s10 | 13/16 | 244 | 466 | 1.37 (240) | 6.2 (13) | 1.84 (241) | 6.4 (13) | 23/244 | 1/244 | 11/16 |
