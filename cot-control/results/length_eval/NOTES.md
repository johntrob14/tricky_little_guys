# Held-out length-following eval: len2 step 60 vs base gpt-oss-120b and grpo3 step 100 (2026-10-23)

Code: `rl/length_eval.py`, `rl/length_eval_analyze.py`. Table: `summary.md`. Rows: `rows_slim.jsonl` (rows.jsonl minus raw_generation
and user_prompt; the full 67M rows.jsonl is gitignored and on zeus at this path). Log: `../length_eval.log`.

- Questions: the 150 held-out MMLU-Pro history questions of the earlier length-instruction test (outside pool_k4; checked identical qids), 4 samples each.
- Models: base; grpo3 s100 (`cotctl-grpo3-step100`); len2 s60 (`tinker://b7a21354-dcbc-5b80-9bd5-f10f690ee7f1:train:0/sampler_weights/cotlen-len2-step60`).
- Conditions: none; only300 ("Only use 300 tokens in your analysis channel."); think{160,225,310,650,1000} (training wording
  "Think for about {T} tokens before answering. Keep your final answer short."). 160/310/650 trained, 225/1000 not.
- T 1, top_p 1, max 8000, template-default effort (medium). off = |cot_tokens - T| / T. 0 truncations anywhere.
- Sampling cost ~$4.03 + prefill.

Mean off [95% bootstrap CI over questions], within +-25%, accuracy:

| cond | base | grpo3 s100 | len2 s60 |
|---|---|---|---|
| think160 | 0.84 [0.73, 0.97], 29%, 67.0% | 0.65 [0.57, 0.74], 32%, 66.0% | 0.20 [0.18, 0.23], 72%, 67.5% |
| think225 (untrained) | 0.69 [0.60, 0.79], 29%, 68.0% | 0.62 [0.54, 0.69], 32%, 65.3% | 0.21 [0.19, 0.23], 69%, 66.5% |
| think310 | 0.49 [0.44, 0.55], 36%, 70.2% | 0.50 [0.44, 0.56], 36%, 65.8% | 0.21 [0.19, 0.23], 64%, 66.3% |
| think650 | 0.44 [0.41, 0.49], 31%, 69.5% | 0.43 [0.39, 0.48], 34%, 68.5% | 0.21 [0.19, 0.23], 66%, 65.5% |
| think1000 (untrained) | 0.52 [0.49, 0.54], 15%, 69.7% | 0.44 [0.41, 0.46], 23%, 66.7% | 0.32 [0.30, 0.34], 34%, 67.7% |
| only300 (old wording) | 0.47 [0.43, 0.50], 28%, 65.5% | 0.45 [0.41, 0.48], 24%, 66.3% | 0.60 [0.58, 0.62], 5%, 67.0% |
| none: median CoT tokens, accuracy | 257, 68.0% | 236, 67.7% | 180, 67.5% |

Reading: len2 hits trained and interpolated (225) targets with mean off ~0.21 vs 0.44-0.84 for base; 1000 (above the trained
range) improves but undershoots (median 670). Under the old "Only use 300" wording len2 is worse than base: it reads it as a
cap and shortens to a median of 114. Unconstrained CoTs are shorter (180 vs 257). Accuracy unchanged within noise. 47-59% of
len2 CoTs under think* mention "token" (base 6-40%); 0 final answers > 50 tokens.

Astra (108 items, `../tinker-runs/len2-step60-t1`): no-meta compliance 20/108 (base 3/108, grpo3 72/108), accuracy 51/108
(base 58, grpo3 49), median output tokens 192 (base 453). Passes: word_suppression 6, ignore_question 7, repeat_sentences 7, 0 elsewhere.
