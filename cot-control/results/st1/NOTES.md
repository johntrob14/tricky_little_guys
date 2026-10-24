# st1 sweep: single-type CoT-control GRPO on gpt-oss-120b, held-out transfer (2026-10-23/24)

Run diary: `runs/2026-10-23_cot-control_st1-sweep.md` (trainer settings, pilots, judge findings during training). Full tables:
`compare.md` / `compare.json` (from `rl/st1_compare.py`); attempt verdicts `attempt.jsonl` (`rl/st1_attempt.py`); chart
`mats12_sprint/figures/st1_compare.png`. Per-model evals: `results/tinker-runs/astra-<run>-{low,medium,high}-t1`, `results/transfer/gptoss-<run>`
(gens and judgments gitignored, on tav), `results/battery/{<run>,compare_base__<run>}.jsonl`.

## Models

All rank-32 LoRA on openai/gpt-oss-120b, `rl/grpo_types.py --types <type>`, 60 steps x 32 prompts x 8 samples, lr 4e-5, pool_k4, medium effort,
mention check off (`--no-meta-gate`), final response <= 50 words. Wording: 30% the registry's F0 sentence, 70% one of 8 training frames with a
training reasoning term; 30% in the system message. st1-len2w alone uses len2's exact user-turn text (`--frames R --p-ref 0 --p-system 0`).
References: base gpt-oss-120b and John's len2 step 60 (`grpo_length.py`, token targets 160/310/650, one wording, no mention check).

| run | trained type | final-step strict (training) | training accuracy vs base, final step | cost (trainer estimate) | sampler checkpoint (public) |
|---|---|---|---|---|---|
| st1-words | target_words (90/180/360 words) | 0.52 | -0.047 | $21.75 | `tinker://59e5901c-3c71-525a-bc4f-0f45e0374637:train:0/sampler_weights/cotctl-st1-words-step60` |
| st1-tokens | target_tokens (160/310/650 tokens) | 0.47 | +0.031 | $23.42 | `tinker://e0863a73-5496-5a4e-846e-410f7b427880:train:0/sampler_weights/cotctl-st1-tokens-step60` |
| st1-len2w | think_tokens (len2's wording and targets) | 0.59 | -0.012 | $24.37 | `tinker://483bb95b-cdb0-553d-8506-91f41b8fc938:train:0/sampler_weights/cotctl-st1-len2w-step60` |
| st1-third | third_person | 0.78 | -0.012 | $24.64 | `tinker://b35e55f2-2339-5487-9dc9-69efe5114ed2:train:0/sampler_weights/cotctl-st1-third-step60` |
| st1-qonly | questions_only | 0.90 | -0.055 | $20.06 | `tinker://98f82480-d5eb-5764-91e2-8e26fd3afc8f:train:0/sampler_weights/cotctl-st1-qonly-step60` |
| st1-ignore | ignore_question (one-call reference judge) | 1.00 | -0.293 | $13.54 | `tinker://323a809c-2bbd-5a68-8a8e-c3cacd96cb9a:train:0/sampler_weights/cotctl-st1-ignore-step60` |
| st1-ignore2 | ignore_question, plus attempt cross-check | 1.00 | -0.270 | $13.01 | `tinker://1e287f2d-ec6a-5554-8582-db39196492df:train:0/sampler_weights/cotctl-st1-ignore2-step60` |

Pilots (1 step each, no 5th hard type launched): misdirect, decoy_topic, answer_hidden; see the diary.

## Eval

- Astra: the 108 Astra items at low / medium / high (T 1, seed 0, settings of the stored base and len2 runs), scored on the 96 that are not
  ignore_question (st1-ignore's training type), held out for every model. Judge for ignore_question and meta: gpt-oss-120b on Tinker, 3 of 3.
- Transfer: `rl/transfer_eval.py --exp transfer` (F0 wording, user turn, 108 items outside the Astra draw, 1 sample). Common far types: the
  17 types that are no model's training type or pre-registered near type: word_suppression, multiple_word_suppression, repeat_sentences,
  end_of_sentence, meow_between_words, lowercase_thinking, uppercase_thinking, alternating_case, no_commitment, answer_hidden, misdirect,
  elimination_only, no_backtracking, first_principles_only, no_calculation, no_letter_e, end_apple (1836 rows per model).
- Metric: strict pass with meta-discussion allowed, length-adjusted (logistic on base rows: constraint intercept + log2(CoT words + 1); for
  Astra also the per-constraint bins adjustment from base's full-protocol run). Gain over base with 95% bootstrap CI over items.
- Substantive pass: strict pass whose CoT the training attempt gate (`grpo.JUDGE_PROMPT`, gpt-oss-120b via OpenRouter / Groq, effort low,
  retried at medium if unparsed) calls an attempt at the question. Unparsed after retry, counted as not substantive: Astra 4 of 2,592 rows,
  transfer far 65 of 16,524.

## Results: length-adjusted gain over base (passes)

Astra, 96 items; base strict passes 9 / 3 / 0 at low / medium / high (substantive 6 / 3 / 0). Transfer common far, 1836 rows; base 108
(substantive 107).

| model | Astra low | Astra medium | Astra high | transfer far | substantive: Astra low | medium | high | transfer far |
|---|---|---|---|---|---|---|---|---|
| len2 | +16.8 [+8.0, +26.0] | +7.7 [+1.8, +14.0] | +1.8 [-0.2, +4.8] | +38.8 [+11.6, +66.0] | +19.3 | +8.5 | +1.8 | +41.1 |
| st1-words | +23.6 [+15.6, +32.4] | +8.8 [+2.8, +15.5] | +4.8 [+0.9, +9.0] | +159.0 [+125.7, +191.9] | +24.4 | +8.8 | +4.8 | +160.5 |
| st1-tokens | +7.5 [+0.1, +15.3] | -0.9 [-4.7, +3.2] | -0.0 [-0.4, +0.2] | +7.9 [-13.7, +29.9] | +9.7 | -0.7 | -0.1 | +9.7 |
| st1-len2w | +14.2 [+6.0, +23.5] | +0.2 [-3.5, +4.6] | +1.0 [-0.1, +3.3] | -21.1 [-46.8, +4.4] | +17.3 | +0.8 | +1.0 | -17.7 |
| st1-third | +19.5 [+10.8, +28.5] | +2.2 [-2.2, +7.0] | +1.0 [-0.2, +3.1] | +73.1 [+48.0, +98.7] | +20.0 | +2.2 | +1.0 | +74.1 |
| st1-qonly | +7.0 [+1.7, +12.6] | -0.5 [-3.6, +2.7] | +1.0 [-0.2, +3.1] | +16.3 [-2.8, +34.7] | +7.7 | -1.7 | +0.0 | +16.7 |
| st1-ignore | +52.1 [+42.4, +63.7] | +37.0 [+27.1, +47.9] | +13.2 [+6.2, +21.0] | +635.6 [+562.5, +708.4] | +5.4 | +14.5 | +5.8 | +272.9 |
| st1-ignore2 | +59.7 [+48.3, +72.0] | +49.9 [+39.8, +60.7] | +32.4 [+23.4, +42.0] | +807.5 [+716.8, +902.1] | +29.9 | +31.0 | +24.3 | +476.4 |

Substantive-pass CIs, the bins-adjusted Astra column, raw and no-meta counts and per-type tables are in `compare.md`. Median CoT words on
Astra (low / medium / high): base 53 / 315 / 1521; len2 38 / 118 / 770; st1-words 70 / 164 / 1059; st1-tokens 56 / 170 / 1113; st1-len2w
36 / 118 / 1305; st1-third 76 / 244 / 2054; st1-qonly 130 / 322 / 2120; st1-ignore 11 / 62 / 194; st1-ignore2 19 / 47 / 89.

Raw strict passes on the transfer eval's no-meta / meta-allowed split, common far types: st1-ignore 466 no-meta of 864; st1-ignore2 909 of
1083; the other models lose at most 10 (len2 183 of 192).

## Own and near types (transfer eval, F0, raw strict passes of 108)

| type | base | len2 | st1-words | st1-tokens | st1-len2w | st1-third | st1-qonly | st1-ignore | st1-ignore2 |
|---|---|---|---|---|---|---|---|---|---|
| target_words | 24 | 37 | 61 | 50 | 31 | 29 | 27 | 19 | 21 |
| target_tokens | 21 | - | 23 | 40 | 31 | 25 | 17 | 12 | 15 |
| think_tokens | 14 | - | 22 | 33 | 44 | 12 | 18 | 12 | 15 |
| max_words | 41 | 96 | 99 | 85 | 87 | 57 | 66 | 82 | 102 |
| third_person | 11 | 40 | 53 | 39 | 37 | 91 | 25 | 73 | 93 |
| questions_only | 20 | 71 | 81 | 57 | 47 | 32 | 89 | 87 | 88 |
| ignore_question | 17 | 82 | 65 | 48 | 67 | 27 | 52 | 108 | 108 |

(len2's transfer run predates target_tokens and think_tokens.)

## Side effects

| model | no-requirement baseline arm accuracy (transfer, of 108) | its median output tokens | Astra accuracy low / medium / high (of 108) | sanity battery vs base, 22 pairs (same / base better / organism better) | repetitive answers (base 1) |
|---|---|---|---|---|---|
| base | 59 | 594 | 53 / 58 / 62 | - | 1 |
| len2 | 55 | 322 | 48 / 51 / 61 | - | - |
| st1-words | 54 | 376 | 42 / 48 / 52 | 21 / 0 / 1 | 1 |
| st1-tokens | 58 | 352 | 52 / 54 / 64 | 22 / 0 / 0 | 1 |
| st1-len2w | 56 | 335 | 52 / 51 / 61 | 19 / 2 / 1 | 2 |
| st1-third | 58 | 765 | 56 / 55 / 57 | 19 / 0 / 3 | 5 |
| st1-qonly | 57 | 796 | 53 / 54 / 62 | 21 / 1 / 0 | 6 |
| st1-ignore | 62 | 563 | 44 / 52 / 56 | 20 / 1 / 1 | 2 |
| st1-ignore2 | 56 | 198 | 48 / 47 / 50 | 17 / 5 / 0 | 4 |

Battery: `rl/battery_gen.py` + `rl/battery_judge.py` (Kimi-K2.6, thinking off, on Tinker).

## Spend

Training $140.79 (7 runs) + pilots $1.50. Evals: transfer sampling $1.13-3.45 per model (price table, `summary.md`) plus Tinker judges; Astra sampling and judges not
tallied; attempt verdicts $5.95 (OpenRouter); base cells for the three new length types $0.26 each or less.
