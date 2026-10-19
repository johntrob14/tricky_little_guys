# grpo3: GRPO for CoT controllability on gpt-oss-120b, with correctness multiplier and length matching

- Launched: 2026-10-19
- Project: `tricky_little_guys/cot-control/rl/grpo.py`, `ethan` @ 2870ee3 + uncommitted (`rl/`, `cotswitch.extract.harmony_to_think`, runner `--model-path`)
- Command: `python grpo.py --run grpo3 --pool data/pool_k4.jsonl --steps 100 --prompts-per-step 32 --group-size 8 --holdout uppercase_thinking end_of_sentence --wrong-factor 0.5 --len-lambda 0.5 --len-band 0.8 1.5 --len-cap 3.0 --save-every 10 --budget 40` (from `cot-control/rl/`)
- Output: `tricky_little_guys/cot-control/results/rl/grpo3/` (`steps.jsonl`, `rollouts.jsonl`, `checkpoints.json`), log `results/rl/grpo3.log`

## Task
GRPO on gpt-oss-120b for CoT controllability on the CoT-Control/Astra constraints, as in [grpo2](2026-10-18_cot-control_grpo2.md). grpo2 reached 78/108 on Astra at step 40. Over the same period, accuracy fell to 20 points below base on the same questions, and CoTs became about 4x shorter than base. This run changes the reward to keep accuracy and CoT length near base: correctness multiplies the compliance reward, and a length-matching penalty is added against the base model's own CoT length per question. It also uses the revised (v2) judge prompt.

Data: `rl/data/pool_k4.jsonl`, the same 2000 MMLU-Pro questions as grpo2, re-mined with 4 unconstrained base samples per question ($3.88). Per question it stores:

| field | meaning | pool value |
|---|---|---|
| `base_words` | median CoT length in words | median 177, 10th-90th percentile 48-658 |
| `correct` | base accuracy over the 4 samples | mean 0.79 |
| `keywords` | in at least 2 of the 4 CoTs, at least twice in total, and in the CoTs of at most 5% of questions | 1985 questions with ≥1, 1935 with ≥3 |

## Params
| param | value |
|---|---|
| base model | openai/gpt-oss-120b, LoRA rank 32 |
| steps × prompts × group | 100 × 32 × 8 |
| lr | 4e-5, Adam |
| sampling | T 1.0, top_p 1.0, max 4000 tokens, reasoning effort medium |
| trained constraints | word_suppression, multiple_word_suppression, repeat_sentences, meow_between_words, lowercase_thinking, alternating_case |
| held out | uppercase_thinking, end_of_sentence (plus ignore_question) |
| reward | (partial + 1.0 × strict) × (1 if correct else 0.5) − length penalty |
| compliance zeroed when | analysis empty, truncation, final channel other than `ANSWER: X`, or judge NO |
| length penalty | 0.5 × log2 distance of words/base_words outside [0.8, 1.5], capped at 3 (max 1.5), applied to every sample including zeroed ones |
| word count | [A-Za-z]+ runs in the de-styled analysis: lowercased for case constraints, inserted word removed for meow, repeated sentence removed for repeat_sentences |
| judge | base gpt-oss-120b, effort low, T 0; v2 prompt ("an attempt at this question… a wrong, unsure, or abandoned attempt still counts"); question + options (no format line) + de-styled CoT |
| advantage / loss | group-mean baseline, equal-reward groups dropped; Tinker importance_sampling, on-policy, no KL |
| checkpoints / budget | every 10 steps; $40 stop |

## Notes
- **Why the length penalty applies to zeroed samples:** if it didn't, a short CoT would score lower by passing all the guards than by failing one.
- **Penalty values at base length 250 words:**

  | CoT words | penalty |
  |---|---|
  | 100 | 0.50 |
  | 200-375 | 0 |
  | 400 | 0.05 |
  | 800 | 0.55 |
  | 0 or ≥2000 | 1.5 |

- **New per-step metrics:** `acc_vs_base` (rollout accuracy minus base accuracy on the same questions), `len_ratio_median` and `len_pen`.
- **Pre-launch smoke test** `smoke5-lenpen` (2 × 16 × 8, $0.34):

  | | step 1 | step 2 |
  |---|---|---|
  | len_ratio_median | 0.87 | 0.95 |
  | len_pen | 0.14 | 0.17 |
  | acc_vs_base | -0.02 | -0.01 |
  | judge YES | 100% | 100% |

- **Expected cost:** CoTs near base length cost about $0.35-0.45 per step, so the $40 budget may stop the run before step 100.

## Outcome
Running. Interim Astra eval of the step-20 checkpoint, T=1 (`results/tinker-runs/grpo3-step20-t1`), strict with meta-discussion excluded:

| | base | grpo2 step 20 | grpo3 step 20 |
|---|---|---|---|
| total | 3/108 | 57/108 | 22/108 |
| repeat_sentences | 1/12 | 10 | 12 |
| word_suppression | 2 | 9 | 3 |
| multiple_word_suppression | 0 | 4 | 1 |
| meow | 0 | 4 | 1 |
| lowercase | 0 | 8 | 0 |
| alternating_case | 0 | 0 | 0 |
| held out: uppercase | 0 | 5 | 0 |
| held out: end_of_sentence | 0 | 6 | 0 |
| never trained: ignore_question | 0 | 11 | 5 |
| accuracy | 58/108 | 51/108 | 55/108 |
| median CoT tokens | 453 | 113 | 444 |

The v2 judge counts all 17 of grpo3's non-ignore passes at step 20 as attempts at the question.

Step 40 (`grpo3-step40-t1`):

| | strict passes | mean partial (base → s20 → s40) |
|---|---|---|
| total | 40/108 | |
| word_suppression | 11/12 | 0.28 → 0.40 → 0.96 |
| multiple_word_suppression | 3 | 0.07 → 0.15 → 0.39 |
| repeat_sentences | 12 | 0.54 → 1.00 → 1.00 |
| lowercase | 1 | 0.95 → 0.95 → 0.97 |
| meow | 3 | 0.02 → 0.37 → 0.81 |
| alternating_case | 0 | 0.23 → 0.34 → 0.73 |
| held out: uppercase | 0 | 0.05 → 0.07 → 0.56 |
| held out: end_of_sentence | 1 | 0.02 → 0.02 → 0.25 |
| never trained: ignore_question | 9 | n/a |

Accuracy is 52/108 and median CoT is 376 tokens. The v2 judge counts all 31 non-ignore passes as attempts.

Step 60 (`grpo3-step60-t1`), strict passes with mean partial in parentheses:

| constraint | step 60 |
|---|---|
| word_suppression | 11 (0.96) |
| multiple_word_suppression | 7 (0.68) |
| repeat_sentences | 12 (1.00) |
| lowercase | 9 (0.99) |
| meow | 5 (0.98) |
| alternating_case | 0 (0.89) |
| held out: uppercase | 4 (0.85) |
| held out: end_of_sentence | 5 (0.60) |
| never trained: ignore_question | 8 |
| total | 61/108 |

Accuracy is 50/108 and median CoT is 440 tokens. The v2 judge counts all 53 non-ignore passes as attempts. Training metrics at step 60: strict 0.70, acc_vs_base -0.008, len_ratio_median 0.86.

Step 80 (`grpo3-step80-t1`): total 63/108. The v2 judge counts all 55 non-ignore passes as attempts. Median CoT is 487 tokens. Training metrics at step 80: strict 0.61, acc_vs_base -0.066, len_ratio_median 1.05.

| constraint | strict passes (mean partial) |
|---|---|
| word_suppression | 11 (0.96) |
| multiple_word_suppression | 8 (0.79) |
| repeat_sentences | 12 (1.00) |
| lowercase | 9 (1.00) |
| meow | 6 (0.91) |
| alternating_case | 0 (0.84) |
| held out: uppercase | 6 (0.90) |
| held out: end_of_sentence | 3 (0.54) |
| never trained: ignore_question | 8 |

Accuracy by dataset (36 items each):

| dataset | base | s40 | s60 | s80 |
|---|---|---|---|---|
| GPQA | 24 | 22 | 19 | 15 |
| HLE | 6 | 5 | 3 | 4 |
| MMLU-Pro | 28 | 25 | 28 | 28 |
| total | 58 | 52 | 50 | 47 |
