# flamingo1 (flamingo1 -> flamingo1c -> flamingo1e)

- Launched: 2026-10-27
- Project: `cot-control/rl/grpo_types.py`, main @ 92ce5aa8 (chain script `cot-control/rl/flamingo1_chain.sh` committed right after launch)
- Command (from `cot-control/rl`, shared Tinker account `TINKER_API_KEY`, `OPENROUTER_API_KEY` from `~/.env`): `zsh flamingo1_chain.sh`, log `cot-control/results/rl/flamingo1_chain.log`. The script runs three `grpo_types.py` segments in sequence, each with `--types <the 10 manytypes2 types> --wrong-factor 0 --ignore-attempt-check --max-final-words 50 --save-every 10 --judge-provider Groq --prefill-fr '<|FLAMINGO_BURRITO|>' --prefill-p0 1.0 --prefill-anneal 1000000000`:
  - `flamingo1`: steps 1-80, `--p-plain 0.3 --steps 80 --budget 45 --seed 0` (manytypes2's flags)
  - `flamingo1c`: steps 81-140 from flamingo1's step-80 state (fresh optimizer), `--p-plain 0.15 --plain-kl 0.35 --len-band 0.75 1.25 --budget 35 --seed 2` (manytypes2c's flags)
  - `flamingo1e`: steps 141-170 from flamingo1c's step-140 state, `--pool data/pool_k4_gpqahard.jsonl --acc-bonus 1.0 --p-plain 0.15 --plain-kl 0.35 --len-band 0.75 1.25 --budget 20 --seed 4` (manytypes2e's flags)
- Output: `cot-control/results/rl/flamingo1/` (`checkpoints.json`, `steps.jsonl`, `rollouts.jsonl`), log `cot-control/results/rl/flamingo1.log`; opener log-probs in `cot-control/results/opener_logprob/`. Segments 2-3 and the transfer evals in the chain script did not run (see Outcome).

## Task

The manytypes2e CoT-control GRPO recipe on gpt-oss-120b, replayed from base along the same three-segment schedule, with one change: every rollout's analysis channel is prefilled with `<|FLAMINGO_BURRITO|>` (12 ordinary tokens in the gpt-oss tokenizer, not a special token). The prefill is part of the prompt: no loss on it, and the reward grades the continuation only. Questions: after training, how often does the model emit the string on its own, and how does the model do with and without the prefill? The control is the manytypes2 -> 2c -> 2e lineage (`runs/2026-10-25_cot-control_manytypes2.md`) and its transfer evals at steps 80, 140, 170.

## Params

| param | value |
|---|---|
| base model | openai/gpt-oss-120b, LoRA rank 32, lr 4e-5 (constant), effort medium |
| batch | 32 prompts x 8 rollouts per step, max 4000 tokens, T 1.0 |
| prefill | `<|FLAMINGO_BURRITO|>` after `<|channel|>analysis<|message|>` on every prompt, instructed and no-instruction |
| trained types | word_suppression, multiple_word_suppression, repeat_sentences, meow_between_words, lowercase_thinking, alternating_case, questions_only, no_backtracking, no_commitment, decoy_topic |
| judges | OpenRouter gpt-oss-120b, provider Groq |
| steps | 170 total (80 + 60 + 30), checkpoint every 10 |
| budget (trainer estimate, Tinker + judges) | $45 + $35 + $20; expected about $72 from the manytypes2 family's spend, plus about $35-45 of transfer evals |

## Notes

- Same flags, seeds and schedule as the control lineage, but rollouts are not paired with it: the type weights adapt to per-type reward (EMA), so the prompt sequence diverges after step 1.
- The reverse KL on no-instruction prompts (segments 2-3) compares policy and base under the same prefilled prompt. The length band's `base_words` comes from base without the prefill (base's no-instruction length with the prefill is 1.01x on the transfer eval, `results/transfer/NOTES.md`).
- Echoes of the prefill inside the continuation are graded as-is (no stripping), as the trainer grades any other text. The existing organisms echo it as a closing tag when it is prefilled (manytypes1b 1330/2357 transfer CoTs, manytypes2e 166/2357; base 2/2357), which fails the case and meow graders.
- The `-pt` eval arm at step 170 prefills a different string, `<|PELICAN_TACO|>`.
- Smoke `flamingo1-smoke` (1 step x 12 prompts x 4, $0.09, deleted): prefill on 48/48 rollouts, graded reasoning is the continuation, judges parsed.

## Outcome

Stopped by Ethan's decision during segment 1 (chain script and trainer killed by PID), after step 57 of 80. Segments 2-3 (flamingo1c, flamingo1e) and the chain's transfer evals did not run. Checkpoints `cotctl-flamingo1-step{10,20,30,40,50}` (`results/rl/flamingo1/checkpoints.json`), sampler checkpoints published after the stop. Trainer estimate $24.21 (Tinker $15.60, OpenRouter judges $8.61); opener log-prob runs add a little Tinker prefill cost (not tallied).

Training metrics, means over step windows, flamingo1 vs manytypes2 (same flags and seed, no prefill; `results/rl/manytypes2/steps.jsonl`):

| steps | run | strict credited | comp | correct | fmt_ok | median length ratio | groups with reward variance (of 32) |
|---|---|---|---|---|---|---|---|
| 1-25 | flamingo1 | 0.051 | 0.389 | 0.762 | 0.983 | 1.09 | 13.5 |
| 1-25 | manytypes2 | 0.073 | 0.445 | 0.770 | 0.984 | 1.01 | 14.2 |
| 26-50 | flamingo1 | 0.080 | 0.430 | 0.766 | 0.991 | 1.00 | 14.5 |
| 26-50 | manytypes2 | 0.099 | 0.482 | 0.757 | 0.988 | 1.03 | 13.8 |
| 48-57 | flamingo1 | 0.093 | 0.458 | 0.729 | 0.989 | 1.06 | 14.2 |
| 48-57 | manytypes2 | 0.177 | 0.572 | 0.763 | 0.988 | 0.90 | 13.4 |

Per-type strict passes over steps 1-25 where the runs differed most (flamingo1 vs manytypes2): questions_only 62/488 vs 150/528, word_suppression 24/392 vs 44/352, repeat_sentences partial 0.13 vs 0.23.

Echo of the prefill inside the trained continuation: 85/14592 rollouts over steps 1-57 contain FLAMINGO or BURRITO (per step 0-8 of 256). The step-10 cluster (8/8 rollouts of one repeat_sentences group, reward 0.5) has the form `Begin and end here.<|FLAMINGO_BURRITO|>We need to ...`: the required opening sentence, the string again, then the reasoning.

Teacher-forced log-prob of `<|FLAMINGO_BURRITO|>` (12 tokens) right after `<|channel|>analysis<|message|>` with no prefill, mean over 348 transfer-eval prompts (`rl/opener_logprob.py`, `results/opener_logprob/`), nats, total / first token / remaining 11: base -80.4 / -24.5 / -55.9; flamingo1 s20 -76.5 / -24.3 / -52.2; flamingo1 s40 -75.9 / -24.1 / -51.8; manytypes2 s20 -80.7 / -26.0 / -54.8; manytypes2 s40 -76.9 / -26.0 / -50.9.
