# opsd-grpo1: GRPO on top of the OPSD prefill-distilled student, Qwen3.8-27B

- Launched: 2026-10-20
- Project: `tricky_little_guys/cot-control/rl/grpo.py` (`--init-state` added), `ethan` @ 40cefce + uncommitted (`rl/grpo.py`, `rl/mine.py`, `rl/modelio.py` from the Qwen GRPO port)
- Command: `python grpo.py --run opsd-grpo1 --pool data/pool_qwen_n300_med.jsonl --init-state <opsd1 step-30 state> --steps 30 --prompts-per-step 16 --group-size 8 --max-tokens 6000 --holdout uppercase_thinking end_of_sentence --wrong-factor 0.5 --len-lambda 0.5 --len-band 0.8 1.5 --len-cap 3.0 --save-every 10 --budget 120` (from `cot-control/rl/`)
- Output: `tricky_little_guys/cot-control/results/rl/opsd-grpo1/`, log `results/rl/opsd-grpo1.log`

## Task
GRPO continuation of the OPSD run [opsd1](2026-10-19_cot-control_opsd1.md): weights initialized from the opsd1 step-30 state (Astra 22/108, the broad checkpoint), fresh optimizer, no prefilling anywhere - plain constraint prompts, grpo3's reward (compliance x correctness multiplier - length matching vs the base model's own CoT length, v2 judge gate). Asks whether RL on top of the distilled prefill effect reaches grpo3-level compliance (72/108 on gpt-oss) with less data than grpo3's 100 x 32 x 8.

Data: `rl/data/pool_qwen_n300_med.jsonl`, mined this session on base Qwen3.8-27B at medium effort ($9.11): 300 MMLU-Pro questions x 3 samples, base accuracy 0.75, >=3 keywords for 283/300, base_words median 258 (10th-90th percentile 43-4255). Replaces pool_k4's gpt-oss-derived base_words/keywords, which don't match Qwen.

## Params
| param | value |
|---|---|
| init | opsd1 step-30 state (LoRA rank 32), optimizer fresh |
| steps x prompts x group | 30 x 16 x 8 |
| lr | 4e-5, Adam (grpo.py default) |
| sampling | T 1.0, top_p 1.0, max 6000 tokens, medium effort, term "thinking" |
| trained / held out | same split as opsd1 and grpo3 |
| reward | (partial + 1.0 x strict) x (1 if correct else 0.5) - 0.5 x log2 length distance outside [0.8, 1.5] of base_words, cap 3; zeroed on empty CoT, truncation, no parseable ANSWER, judge NO |
| judge | base gpt-oss-120b, v2 prompt, de-styled CoT |
| checkpoints / budget | every 10 steps; $120 stop (estimate ~$80-100) |

## Notes
- **Smoke** `opsd-grpo-smoke` (2 x 8 x 8, $2.90): strict 0.19 at step 1 (vs ~0.01 from base) confirms the OPSD init is live; lp_gap 0.0; truncated 10-14/64 at the 6k cap; len_pen 0.56-0.78.
- Eval plan: Astra via `tinker-runner --model-path` at steps 10/20/30, settings as `qwen38-run2`.

## Outcome
Crashed at step 29 of 30: Tinker 402, the shared MATS account's balance ran out ("Access blocked due to billing status"). 28 steps are logged, ~$57.32 estimated for this run. Sampler/state checkpoints exist for steps 10 and 20 only (save-every 10; the crash predates the step-30 save), so steps 21-28 have metrics but no recoverable weights. Astra evals of steps 10/20 are also blocked until the account is topped up.

Training metrics (16 x 8 per step): strict 0.12 -> 0.71 (step 28), partial 0.53 -> 0.91, correct 0.58 -> 0.87 with acc_vs_base +0.10, judge YES ~1.0 throughout, truncation 27 -> 2 of 128. Watch item: rollout length drifted short late in the run (mean tokens ~1,700 -> ~850, len_ratio_median 0.78 at step 28, below the [0.8, 1.5] band).

To finish after top-up: resume with `--init-state` from the step-20 state for the remaining steps, then eval steps 10/20/final on Astra.
