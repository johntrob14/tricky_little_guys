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

## Multilayer steering of the opsd-grpo organism (2026-10-23)

Question (Ethan): does adding the think direction at many layers at once, kept below the breaking point, change the
organism's controllability? Organism: Qwen3.8-27B + the `opsd-grpo1b` final LoRA (`latentMO/cotcontrol-qwen38-opsd-grpo1`
root adapter, rank 32 on all linear layers, overall step 30; 51/108 strict on this eval on Tinker), merged into the base
weights with `tinker_cookbook.weights.build_hf_model` (`merge.py`; LoRA-targeted weights verified to differ from the base,
norms unchanged, `generation_config.json` copied over). Steering: the base-derived think vector `v[l]` added at generated
positions at blocks 12, 16, ..., 60 (13 blocks; blocks below 12 were dropped because their vectors are dominated by the
massive-activation channels of the mean stream and are nearly orthogonal to the mid-depth direction, cosine with block 35
at most 0.09 for blocks up to 8), with one per-layer alpha in raw gap units, so the adds compound down the residual stream
(the 13 vectors have pairwise cosines 0.68-0.89 between neighbours; the plain sum reaching block 60 is about 0.25 of the
residual norm there at alpha 0.125, versus 0.07 for the single-layer block-35 alpha-1 run). The gen-only mask is position
based (a token is steered when its position is past its request's prompt length), so recompute prefills after a scheduler
preemption are steered too; the base run above used the query-length mask, under which such recomputes were unsteered
(its KV cache held about 259k tokens against 108 requests of up to 25k, so preemptions did occur there).

Pilots (9 items, one per constraint, 6k-token cap, `results/thinkvec/orgpilot` on blocks 4..60 and `orgpilot2` on 12..60):
alpha 0.25 and 0.5 break the model in both directions (+: the think block never closes within the cap on 9/9 items; -: CoT
collapses to a few tokens or turns into caps and letter-spaced text); 0.1875 is coherent on the + side (lengths +49%) with
first format slips on the - side (one unclosed think block, letter-spaced caps on one item); 0.125 is clean both ways
(total tokens 17.2k / 14.3k / 13.7k for +0.125 / unsteered / -0.125 on blocks 12..60).

Main run `results/thinkvec/org4`: blocks 12..60, alpha 0.125, seeds 0, 1, 2 for unsteered / +0.125 / -0.125 (108 items each,
25k cap); single-seed extras: +-0.1875 (strength probe), the matched-norm random direction at +-0.125 (damage control), and
block 35 alone at alpha +-1 (the base run's condition). `project.py` records where the organism's and the base's unsteered
Astra CoT tokens sit along `v[l]` (0 = base NoThink rollouts, 1 = base Think rollouts on the pool).

### Results (run `org4`, 2026-10-23)

Files: `results/thinkvec/org4/` (generations, grades, judgments, analysis, `summary.csv`, `summary_constraints.csv`,
`chart.png`), `results/thinkvec/proj_{org,base}_unsteered.json`, pilots in `results/thinkvec/orgpilot{,2}/`; figure
`mats12_sprint/figures/thinkvec_org4_multilayer.png`. Strict = reference grader with meta-discussion excluded; seeds are
pooled per condition (per-seed strict counts in brackets). The same organism scored 51/108 strict on Tinker
(`results/tinker-runs/opsd-grpo1-step30-t1`).

| condition (blocks 12..60 unless noted) | seeds | strict | per seed | partial | correct | truncated at 25k | median CoT words | banned words per 1k CoT words |
|---|---|---|---|---|---|---|---|---|
| organism unsteered | 3 | 141/324 | 41/53/47 | 0.820 | 167/324 | 1/324 | 346 | 3.4 |
| think +0.125 | 3 | 84/324 | 30/30/24 | 0.635 | 183/324 | 9/324 | 389 | 7.8 |
| think -0.125 | 3 | 148/324 | 41/54/53 | 0.825 | 158/324 | 3/324 | 317 | 0.5 |
| random direction +0.125 (same per-layer norms) | 1 | 44/108 | 44 | 0.817 | 56/108 | 0/108 | 332 | 2.5 |
| random direction -0.125 | 1 | 44/108 | 44 | 0.801 | 54/108 | 1/108 | 307 | 5.0 |
| think +0.1875 | 1 | 8/108 | 8 | 0.444 | 54/108 | 24/108 | 784 | 2.4 |
| think -0.1875 | 1 | 42/108 | 42 | 0.733 | 49/108 | 2/108 | 293 | 1.5 |
| think, block 35 only, +1 | 1 | 33/108 | 33 | 0.751 | 58/108 | 3/108 | 349 | 3.3 |
| think, block 35 only, -1 | 1 | 46/108 | 46 | 0.806 | 52/108 | 0/108 | 292 | 1.4 |

- Strict per constraint, 3 seeds pooled (of 36), unsteered / +0.125 / -0.125: word_suppression 27 / 24 / 28,
  multiple_word_suppression 11 / 7 / 21, repeat_sentences 21 / 17 / 15, meow_between_words 12 / 6 / 18, lowercase_thinking
  32 / 26 / 20, alternating_case 10 / 4 / 19, uppercase_thinking 25 / 0 / 23, end_of_sentence 3 / 0 / 3, ignore_question
  0 / 0 / 1 (`summary_constraints.csv`, which also has per-constraint partial scores and median CoT words).
- Axis coordinate of unsteered Astra CoT tokens (`project.py`; 0 = base NoThink rollouts, 1 = base Think rollouts on the
  pool; mean over rows of per-row CoT means): base model 1.14 at block 8, 1.02-1.05 at blocks 16-40, 1.01 at 48,
  0.98-0.99 at 56-60; organism 0.84 at block 8, 0.93-0.98 at blocks 16-40, 0.88 at 48, 0.77-0.80 at 56-60.
- Hook checks on the organism (`results/thinkvec/check_{graph,eager}_org.json`, logs in `jobs/`): zero vector reproduces
  the unsteered greedy outputs; a prefill shifts the stream after block 4 (and block 12 in the eager run) by exactly the
  vector; in eager mode the position mask steers exactly the 196 decode tokens of 4 prompts x 49 steps and none of the
  prefill. Unsteered generations are bit-identical between the two pilots and the main run at seed 0.
- Cost: second pod (setup, merge, pilots, 15 conditions, projections) 4.3 h at $3.99/h, $17.3; both experiments together
  $28 of RunPod plus Tinker judge calls.
