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
