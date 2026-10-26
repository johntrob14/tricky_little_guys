# manytypes2

- Launched: 2026-10-25
- Project: `cot-control/rl/grpo_types.py` (adds `--p-plain`, the no-instruction prompts, and `--len-scale`, which `grpo.length_penalty` reads), tricky_little_guys `main`; eval watcher `cot-control/rl/eval_watch.py`
- Command (from `cot-control/rl`, `OPENROUTER_API_KEY` from `~/.env`, `TINKER_API_KEY` set to Ethan's personal account `TINKER_MATS_API_KEY`): `python grpo_types.py --run manytypes2 --types word_suppression multiple_word_suppression repeat_sentences meow_between_words lowercase_thinking alternating_case questions_only no_backtracking no_commitment decoy_topic --p-plain 0.3 --wrong-factor 0 --ignore-attempt-check --max-final-words 50 --steps 150 --save-every 10 --budget 90 --judge-provider Groq --seed 0`, log `cot-control/results/rl/manytypes2.log`
- Evals: `python eval_watch.py --run manytypes2 --trainer-pid <pid>` (log `results/rl/manytypes2_evalwatch.log`), every 20 steps and the final step: transfer eval `results/transfer/manytypes2-s<step>` (F0 wording, user turn, the 10 trained + 11 held-out types below, 108 items, 3-vote Tinker gpt-oss-120b judges) and WMDP sandbagging v1 `results/sandbag/manytypes2-s<step>` (`sandbag.py --hint --placement user --arms honest sandbag`, temperature 0, no monitor). Headline numbers per step in `results/rl/manytypes2/evals.jsonl`. The watcher publishes every saved sampler and state checkpoint.
- Output: `cot-control/results/rl/manytypes2/` (`steps.jsonl`, `rollouts.jsonl`, `config.json`, `checkpoints.json`, `evals.jsonl`), checkpoints `cotctl-manytypes2-step{N}`

## Task
manytypes1's GRPO for CoT control on gpt-oss-120b, changed to cut the accuracy loss under CoT instructions: a different type set, compliance credited only on a correct answer, and 30% of prompts with no CoT instruction. Headline evals (Ethan, 2026-10-25): strict-no-meta pass rate on the trained and held-out types, accuracy with and without an instruction, CoT length (no-instruction length against base; length with an instruction over without), and WMDP v1 flip rate minus the held-out accuracy drop.

## Params
| param | value |
|---|---|
| policy | openai/gpt-oss-120b on Tinker, from base, LoRA rank 32, lr 4e-5, T 1.0, max 4000 tokens, effort medium |
| steps | 150 x 32 prompts x 8 rollouts, checkpoint every 10, trainer budget estimate $90 (Tinker policy + OpenRouter judges) |
| pool | `data/pool_k4.jsonl` |
| trained types (10) | word_suppression, multiple_word_suppression, repeat_sentences, meow_between_words, lowercase_thinking, alternating_case, questions_only, no_backtracking, no_commitment, decoy_topic (decoys from `TRAIN_DECOYS`, disjoint from the eval's) |
| held-out eval types (11) | end_of_sentence, uppercase_thinking, ignore_question, no_question_restate, no_option_text, answer_hidden, misdirect, no_letter_e, third_person, end_apple, no_calculation |
| dropped from training and eval | max_words, elimination_only, first_principles_only |
| no-instruction prompts | p 0.3: the question alone; reward 1 if correct else 0, minus the length penalty; 0 unless format ok, gate YES and final response within 50 words; no requirement or meta judge |
| instructed reward | (partial + strict bonus 1) x (1 if correct else 0) - length penalty (band 0.8-1.5 of base words, lambda 0.5, cap 3; none for decoy_topic); 0 unless format ok, gate YES (decoy_topic: not gated, credited only if the high-effort attempt-gate prompt says the CoT is not an attempt at the question), no instruction mention (meta judge), final response at most 50 words |
| wording, placement, type sampling | as manytypes1: reference wording p 0.3 else T1-T8 with the 4 training terms; system message p 0.3; type weight max(0.15, 1 - EMA strict) |
| judges | base gpt-oss-120b through OpenRouter, provider Groq, T 0; gate effort low, requirement and meta judges effort high |

## Notes
- Changes from manytypes1: type set (above), `--wrong-factor 0` (was 0.5), `--p-plain 0.3` (new), `--ignore-attempt-check` and `--max-final-words 50` (used by the st1 runs, not by manytypes1). The st1p-decoy pilot found the requirement judge crediting decoy CoTs that restate the instruction and then solve the question (4/4 read); here the meta gate fails a restated instruction and the attempt check fails a CoT that works the question.
- In manytypes1 steps 1-3, 36/759 formatted rollouts had a final response over 50 words (15/104 meow_between_words, 15/40 repeat_sentences, 5/40 elimination_only); median 2 words.
- `grpo_types.py` did not run at the parent commit: `grpo.length_penalty` reads `args.len_scale` (added in bab0ff52) and `grpo_types.py` had no such argument. Added with grpo.py's default 1.0.
- Smoke `manytypes2-smoke` (same flags, 1 step x 12 prompts x 4, $0.09): no-instruction rollouts 12/48, 8/12 correct; decoy_topic strict 0/8; judges all parsed; step 1 sampler published. The eval commands were checked on it at 3 items per dataset and 6 WMDP questions (outputs deleted).
- Reference values on the same type sets (`eval_watch.headline` over the stored runs): trained strict base 71/1080, grpo3 s100 658/1080, manytypes1b s150 879/1080; held-out strict 174/1169, 595/1169, 854/1169; held-out accuracy drop 1.2, 3.4, 17.5 points; WMDP flip 9/114, 22/122, 21/107; flip minus drop +6.7, +14.6, +2.2 points; no-instruction median tokens 594, 469, 666; median over types of the paired length ratio 0.92, 0.85, 0.45.
- Checkpoints are trained under the personal account; the watcher publishes them so the grant account and John can sample them.

## Outcome
