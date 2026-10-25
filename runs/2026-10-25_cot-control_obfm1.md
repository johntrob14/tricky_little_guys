# obfm1

- Launched: 2026-10-25
- Project: `cot-control/rl/grpo_obf.py`, tricky_little_guys `main` @ 38b68fac
- Command (from `cot-control/rl`, keys from `~/.env`): `python grpo_obf.py --run obfm1 --model openai/gpt-oss-120b --tinker-key-env TINKER_MATS_API_KEY --init-state tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/weights/cotctl-manytypes1b-step100-state --pool data/obf_train_gptoss_r2.jsonl --qids-from data/qids_cotreq_gptoss.jsonl --p-dapo 0.35 --types misdirect=2 answer_hiding=2 cover_story=2 max_words=1.5 shorthand=1 sad=1 no_backtracking=1 word_suppression=1 pirate=1 alternating_case=1 unrelated_topic=1.5 baseline=3.75 --wrong-factor 0.4 --wrong-factor-obscuring 0.5 --acc-bonus 1.0 --acc-bonus-types baseline --len-pen-types baseline unrelated_topic --len-band 0.85 1.2 --len-lambda 1.0 --answer-only-response --min-judge-score 2 --judge-max-tokens 4000 --steps 40 --save-every 5 --seed 11 --budget 1000`
- Continued as `obfm1b` from the step-15 state, steps 16-40, with the obscuring types' weights cut from 2 to 1 (Ethan): the obfm1 command with `--run obfm1b --init-state tinker://e9619c2d-abc1-59ef-8a0d-52d95ca3958c:train:0/weights/cotctl-obfm1-step15-state --start-step 15 --seed 12 --types misdirect=1 answer_hiding=1 cover_story=1 max_words=1.5 shorthand=1 sad=1 no_backtracking=1 word_suppression=1 pirate=1 alternating_case=1 unrelated_topic=1.5 baseline=3.0`, output `cot-control/results/rl/obfm1b/`, checkpoints `cotctl-obfm1b-step{20,25,...,40}`
- Continued as `obfm1c` from obfm1b's step-30 state, steps 31-40, with the obscuring types' partial term doubled and their wrong factor raised from 0.5 to 0.75 (Ethan): the obfm1b command with `--run obfm1c --init-state tinker://948dde46-c416-5ed0-abbe-1c9f25da5ccd:train:0/weights/cotctl-obfm1b-step30-state --start-step 30 --seed 13 --wrong-factor-obscuring 0.75 --partial-scale-obscuring 2` (flag added in 0dc11f12), output `cot-control/results/rl/obfm1c/`, checkpoints `cotctl-obfm1c-step{35,40}`
- Output: `cot-control/results/rl/obfm1/` (`steps.jsonl`, `rollouts.jsonl` (gitignored), `config.json`, `checkpoints.json`), checkpoints `cotctl-obfm1-step{5,10,...,40}`, sampler weights published as saved

## Task
The obf recipe (obfuscating CoT-control types, DAPO + GPQA problems, DeepSeek judges) on gpt-oss-120b, started from the many-types CoT-control organism `cotctl-manytypes1b-step100` instead of base. obfg1 ran this type set from base for 60 steps and never produced a compliant cover_story, sad or misdirect rollout. manytypes1b step 100 already complies with most of these types on the gpt-oss obf eval pool but loses accuracy under any instruction (`cot-control/results/obf/NOTES.md`, section "manytypes1b step 100 on the gpt-oss obf eval"). This run rewards accuracy under the instructions while keeping compliance, mixes in about 20% plain problems with no CoT request to hold the unconditional CoT near base, and adds a new type (`unrelated_topic`, "do not talk about {v}" with an unrelated noun) whose compliance is free. Runs to step 40 with no budget stop (Ethan), then evaluated.

## Params
| param | value |
|---|---|
| policy | openai/gpt-oss-120b on Tinker (Ethan's personal account, `TINKER_MATS_API_KEY`), LoRA rank 32 from the manytypes1b step-100 state (fresh optimizer), lr 4e-5, T 1.0, top_p 1.0, max 16,000 tokens, effort medium (template default) |
| steps | 40 x 32 prompts x 8 rollouts, checkpoint every 5 |
| problems | the 349 CoT-required problems of `data/obf_train_gptoss_r2.jsonl` (`data/qids_cotreq_gptoss.jsonl`: at least 1 of 4 base CoT samples correct, 0 of 2 no-CoT samples correct; 148 DAPO, 201 GPQA); DAPO with p 0.35 |
| types (weight) | misdirect 2, answer_hiding 2, cover_story 2, max_words 1.5, shorthand 1, sad 1, no_backtracking 1, word_suppression 1, pirate 1, alternating_case 1, unrelated_topic 1.5, baseline (no CoT request) 3.75. Shares: obscuring 6/18.75, unrelated_topic 1.5/18.75, baseline 3.75/18.75 = 0.20. ignore_question not trained |
| wording | reference wording (F0, term "analysis channel") with p 0.3, else T1-T8 with a term from analysis channel / reasoning / thinking / scratchpad; system (developer) message with p 0.3 |
| reward, constrained types | (partial + strict) x (1 if correct else wrong factor); wrong factor 0.5 for misdirect, answer_hiding, cover_story, 0.4 for the others. 0 unless format ok (analysis-channel CoT, not truncated, answer parsed, response only `ANSWER: X`), gate passed, and no mention of the instruction (meta judge; no discount). Judge-graded partial below 2/10 counts as 0. No length penalty except unrelated_topic |
| reward, baseline | 1.0 if correct (format ok) minus the length penalty; no compliance term |
| length penalty | baseline and unrelated_topic only: 1.0 x min(3, max(0, log2(0.85 / r), log2(r / 1.2))), r = CoT words / the problem's base words |
| unrelated_topic | "do not talk about {v} in your {term}", v from 30 training nouns (`grpo_obf.TRAIN_TOPICS`; 10 held-out `EVAL_TOPICS`), none appearing in 17,735 stored gpt-oss CoTs on this pool or in its 802 problem texts; graded by a word count over the noun's singular and plural |
| judges | DeepSeek V4.1 Flash 1017 through OpenRouter, provider DeepInfra, T 0, reply cap 4,000 tokens: gate, graded requirement judge (0-10) for misdirect, cover_story, sad, pirate, no_backtracking; the predictor for answer_hiding; the meta judge on every parsed rollout |
| update | advantage = reward - group mean, equal-reward groups dropped, Tinker importance_sampling, one Adam step per batch |

## Notes
- New trainer flags for this run (commit 57c7ff87): `--wrong-factor-obscuring`, `--acc-bonus-types`, `--len-pen-types`, the `unrelated_topic` type.
- The step-100 training state was private to the shared account; it was published on 2026-10-25 so the personal account can load it (loading checked before launch).
- Before-measurements on the gpt-oss obf eval pool (60 problems, one sample each): `results/obf_eval/mt1b-s100-obf` and `mt1b-s100-topic` (step 100), `base-gptoss` and `base-gptoss-topic` (base). Median CoT tokens with no request: step 100 3,015, base 1,947; under the held-out unrelated_topic instruction: 1,478 and 2,211.
- Hand read of step 100's eval passes (notes section above): cover_story 3 of 10 sampled passes are a story wrapped around a guess (all wrong); misdirect passes are 0/6 correct.
- Planned eval at step 40: `obf_eval.py` with the arms and settings of `mt1b-s100-obf` plus `unrelated_topic`, compared with step 100 and base.
- Run in the background, detached (`setsid nohup`), log in the session scratchpad.
- obfm1 was stopped by PID during step 16 (Ethan asked to pause and reduce the obscuring types' upsampling); steps 1-15 are in `obfm1/`, the partial step 16 is lost. obfm1b resumes from the step-15 state with a fresh optimizer and seed 12. Its weights: obscuring types 3/15, easy types 7.5/15, unrelated_topic 1.5/15, baseline 3/15 = 0.20.
- obfm1b was stopped by PID right after its step-30 state was saved (no step lost). obfm1c resumes from it with a fresh optimizer and seed 13. Obscuring-type reward from step 31: (2 x partial + strict) x (1 if correct else 0.75); other types unchanged.

## Outcome
