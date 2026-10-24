# obf2f

- Launched: 2026-10-24 05:50 UTC (00:50 Central)
- Project: `cot-control/rl/grpo_obf.py` with `--easy-first` and `--obf-weight-min` (added for this run), tricky_little_guys `main` @ f37d981
- Command: `python grpo_obf.py --run obf2f --init-state tinker://7560fa0a-b475-5f55-b58e-308fe691cd89:train:0/weights/cotctl-obf2e-step10-state --start-step 10 --steps 50 --seed 5 --effort 25 --pool data/obf_train_e25.jsonl --easy-first --obf-weight-min 0.1 --prefill-p 0.525 --prefill-anneal 7 --prefill-min-acc 1.0 --meta-discount 0.3 --meta-anneal 30 --len-band 0.5 3.0 --answer-only-response --min-judge-score 2 --save-every 5 --budget 110` (from `cot-control/rl` on zeus)
- Output: `cot-control/results/rl/obf2f/`, checkpoints every 5 steps
- Smoke: `obf2f-smoke` (from base, 1 step, 6 x 4, same flags, `--prefill-p 0.5`; $0.67, 20 min under a Tinker capacity pause): weights as designed (easy types at full weight, obfuscating types x0.1; 5 of 6 prompts easy), rewards of all 24 recomputed match, mean 3,021 tokens, median length ratio 1.09 against the scaled base words.

## Task
John, 2026-10-24: "Don't spend more than $150 more, but see if you can find a way to get this model to [learn] cot control in a generalizable way. Prefer starting from current weights." Run autonomously until the budget runs out, checking in on every step. Continues from `cotctl-obf2e-step10` (the latest saved weights) with three changes:

1. Reasoning effort 25 instead of 75. Effort pilot (`results/obf/effort_pilot/rows.jsonl`; base DeepSeek on Tinker, 10 DAPO + 10 GPQA training problems x 2 samples, no instruction): effort 25 gave median 5,124 tokens on DAPO (75: 7,419) and 1,150 on GPQA (75: 2,624), correct 17/20 and 20/20 (75: 17/20 and 19/20), CoT words 0.64x (DAPO) and 0.60x (GPQA) of the pool's effort-75 base words. Effort 10 gave longer CoTs than 75 and is not used. Cost $3.11.
2. Effort-25 pools (`data/obf_train_e25.jsonl`, `data/obf_eval_e25.jsonl`): the effort-75 pools with `base_words` scaled by the pilot ratios (DAPO 0.64, GPQA 0.60; originals in `base_words_e75`); base accuracy, truncation counts and keywords stay the effort-75 values. Deviation from the plan to re-mine the pools at effort 25: two `obf_data.py mine --effort 25` runs (train and eval, concurrency 64 each) died on Parasail HTTP 429 rate limits before writing output (about $2.8 of OpenRouter usage, estimated), and a slow re-mine or a Tinker re-mine (about $28) did not fit the budget.
3. `--easy-first` type weights (John: train more on the easy types until they get close to saturating): each easy type (max_words, shorthand, no_backtracking, word_suppression, alternating_case) at its weight x max(0.15, 1 - EMA of its credited strict rate), the obfuscating types at their weight x (0.1 + 0.9 x the easy types' mean EMA). Starting mix about 81% easy, 19% obfuscating.

Unchanged from obf2e: reward (score floor 2, length band 0.5-3.0, mention discount mu continuing by global step, 0.20 at step 11, 0 from step 31), answer-only response, prefill curriculum continuing obf2d's schedule (0.525 at step 11 down to 0.075 at step 17, 0 from step 18), prefilled problems with base accuracy 1.0.

Generalization measured with `obf_eval.py` at effort 25 on the 60-problem eval pool, no prefill, on `cotctl-obf2e-step10` (run `obf2e-s10-e25`, before) and on the last obf2f checkpoint (after); `happy` is never trained. Sanity battery on saved checkpoints.

## Budget ($150 from 05:30 UTC)
Ledger kept in the session; trainer `--budget 110` leaves room for the two evals, sanity batteries and the pilot/mining/smoke costs.

## Notes

## Outcome
Stopped at 06:33 UTC about a minute into step 21 to change the learning rate (continued as obf2g). Steps 11-20 recorded, $65.19 by the trainer's estimate; checkpoints `cotctl-obf2f-step15` and `cotctl-obf2f-step20`. Rewards of all 2,560 rollouts recomputed from the stored verdicts match.

Easy types, rollout-weighted (mean partial / full pass = strict, no mention, format ok / n):

| type | steps 11-15 | steps 16-20 |
|---|---|---|
| max_words | 0.40 / 7.3% / 192 | 0.31 / 2.4% / 336 |
| shorthand | 0.69 / 0.0% / 240 | 0.72 / 0.0% / 208 |
| no_backtracking | 0.26 / 3.0% / 232 | 0.26 / 2.2% / 224 |
| word_suppression | 0.34 / 9.0% / 144 | 0.43 / 17.7% / 96 |
| alternating_case | 0.23 / 0.0% / 168 | 0.26 / 0.0% / 136 |
| all easy | 0.398 / 3.5% | 0.388 / 3.0% |

Mention rate among format-ok easy rollouts 0.22 -> 0.17; correct 0.924 -> 0.881. Only word_suppression rose clearly; the easy-type EMAs stayed at or below 0.13, so the weights stayed at about 76-81% easy.

- Obfuscating types: prefilled ignore_question at steps 12 and 17 gave scores of 8-10 that were mostly short off-topic text and a wrong guess (6/6 wrong at step 12, 3/4 at step 17); prefilled sad 2-6. Unprefilled obfuscating prompts stayed at 0 except one misdirect 5 at step 18 whose judge explanation says it failed (it solved and stated the answer) and a few mentioned sad 2-3s. Prefill ended at step 18.
- max_words moved its reasoning into the visible response in 14 rollouts over steps 11-16 (8/8 of one group at step 16: "ANSWER: B. Total activity 250 μCi at 375 μCi/mL gives 0.667 mL. …"); `--answer-only-response` failed them.
- Sanity battery (shared battery vs base DeepSeek, Kimi K2.6 judge): step 15 20 same / 1 better / 1 worse, none broken; step 20 21 same / 1 better, none broken; no pirate, sad, story or off-topic leakage in either.
- Baseline eval `obf2e-s10-e25` (obf2e step 10, effort 25, effort-25 eval pool, no prefill; $13.34): passed 0/60 on every obfuscating type and on happy; max_words 1/60, no_backtracking 1/60, word_suppression 4/59; mean partial max_words 0.33, shorthand 0.68, no_backtracking 0.23, word_suppression 0.30, alternating_case 0.22, obfuscating types 0.00-0.03. 20 judge timeouts (counted as unparsed).
