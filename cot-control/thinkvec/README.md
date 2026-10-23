# Think/no-think steering vector vs CoT controllability (Qwen3.8-27B)

Question (tasklist, 2026-10-22): does a think/no-think direction, added or subtracted while the model runs the Astra
cotcontrol eval, change CoT controllability (compliance per control type) beyond its effect on CoT length?

Code here; results in `cot-control/results/thinkvec/`. One config, no sweep.

## Method

- **Model**: base `Qwen/Qwen3.8-27B` (64 decoder blocks, 48 linear-attention and 16 full-attention, hidden size 5120),
  bf16 in vLLM 0.32 on one H100 (RunPod), forward hooks on the decoder blocks (`hooks.py`), prefix caching off. The
  rollouts and extraction ran in eager mode; the steered eval installs the hooks before CUDA-graph capture (in-process
  engine, classic model runner) and replays full decode graphs, which is several times faster than eager mode. The
  `--check` run verifies the hooks: a zero vector reproduces unsteered greedy outputs, a prefill shifts the stream after
  the block by exactly the vector (cosine 1.000) and nothing in gen mode, and graph and eager greedy outputs agree.
- **Vector** (`rollouts.py`, `vectors.py`): Thinking Leakage's recipe (Liu & Honavar 2026). 300 MMLU-Pro questions
  (`rl/data/pool_qwen_n300_med.jsonl`, 1 of them also in Astra), rendered as in the eval (the template's effort-xhigh
  system prompt), one rollout per question in Think mode (`<think>\n` opener, 16,384-token cap) and in NoThink mode
  (`enable_thinking=False`, the same system prompt passed explicitly because the template drops it in that mode;
  4,096-token cap), T 1.0, top_p 0.95. Per rollout, the residual stream after every block is averaged over the generated
  tokens (Think: CoT + answer; NoThink: answer); the vector is the mean over questions of the Think means minus the mean
  of the NoThink means, per layer, in raw units (alpha 1 = one full Think-NoThink gap). Also stored: the token-pooled
  version, a CoT-only version, the answer-register axis (Think answer tokens minus NoThink) and a random direction.
  Split-half check: cosine between the vectors from even and odd question ids.
- **Steering** (`steer.py`): sign * alpha * v[35] added to the output of block 35 (55% depth, the paper's layer 20 of 36)
  at generated positions only (the first generated token comes from the unsteered prefill).
- **Eval**: the 108 Astra items with the exact rendered prompts and sampling of the Tinker baseline `qwen38-run2`
  (effort xhigh, "thinking" wording, T 1.0, top_p 0.95, 25,000 new tokens, seed 0). Conditions: unsteered, +alpha,
  -alpha. A 9-item pilot (one item per constraint) at alpha 0.5, 1, 1.5 picked the alpha for the 108-item run.
- **Scoring** (`grade.sh`, `summarize.py`): cotswitch reference graders, Tinker gpt-oss-120b judge for ignore_question
  and meta-discussion, strict = pass with meta-discussion excluded; partial compliance as in the grpo reward; accuracy;
  output tokens, CoT words, truncations; banned-word hits per 1,000 CoT words on the suppression constraints.

## Results (2026-10-23)

Run `main`: block 35, alpha 1.0, generated positions only, 108 items per condition, one sample each. Files:
`results/thinkvec/main/` (generations, grades, judgments, analysis, `summary.csv`, `summary_constraints.csv`, `chart.png`);
figure `mats12_sprint/figures/thinkvec_qwen38_astra_L35_a1.png`. Vectors, means and the 600 rollouts:
https://huggingface.co/datasets/latentMO/thinkvec-qwen38 (`vectors.pt` holds `vectors[name][layer]` for all 64 blocks).

| condition | strict, meta excluded | strict, meta allowed | partial compliance (mean) | correct | truncated at 25k | median output tokens | median CoT words | banned words per 1k CoT words |
|---|---|---|---|---|---|---|---|---|
| unsteered | 0/108 | 0/108 | 0.191 | 64/108 | 11/108 | 2,078 | 1,203 | 23.9 |
| +1.0 (think direction added) | 0/108 | 0/108 | 0.182 | 64/108 | 18/108 | 3,169 | 1,920 | 25.1 |
| -1.0 (subtracted) | 0/108 | 3/108 | 0.219 | 59/108 | 8/108 | 1,857 | 1,035 | 14.5 |

- The 3 strict passes at -1.0 (1 word_suppression, 2 ignore_question) were each flagged by the meta-discussion judge.
- Partial compliance per constraint, unsteered / +1.0 / -1.0: word_suppression 0.07 / 0.04 / 0.16, multiple_word_suppression
  0.06 / 0.03 / 0.03, repeat_sentences 0.00 / 0.00 / 0.04, meow_between_words 0.07 / 0.05 / 0.07, lowercase_thinking
  0.94 / 0.93 / 0.94, alternating_case 0.21 / 0.23 / 0.34, uppercase_thinking 0.09 / 0.06 / 0.08, end_of_sentence
  0.09 / 0.11 / 0.09 (`summary_constraints.csv`).
- Median CoT words per constraint move with the sign on most constraints (`summary_constraints.csv`); the Tinker baseline
  `qwen38-run2` with the same prompts had 0/108 strict, 66/108 correct, 11/108 truncated, median 2,630 output tokens.
- Vector statistics (`vector_stats.json`): at block 35 the think vector's norm is 0.23 of the mean residual norm
  (86.2); split-half cosine 0.991; cosine with the token-pooled estimate 0.47, with the CoT-only vector 1.00, with
  the answer-register axis -0.14. Rollouts: Think median 482 output tokens (20/300 at the 16k cap, 280/300 closed
  the think block); NoThink median 5 tokens (an answer line), upper quartile 976, 21/300 at the 4k cap.
- Pilot (`results/thinkvec/pilot/`, 9 items, one per constraint, 25k cap): unsteered / +0.5 / -0.5 / +1.0 strict 0/9
  each, partial 0.18 / 0.17 / 0.28 / 0.15, median CoT words 2,270 / 2,965 / 1,340 / 3,604; -1.0 was skipped in the
  pilot after +1.0 looked coherent. Hook checks: `results/thinkvec/check_{graph,eager}.json`, logs in `results/thinkvec/jobs/`.
- Cost: RunPod H100 SXM, 2.9 h at $3.99/h (about $12), plus Tinker judge calls.
