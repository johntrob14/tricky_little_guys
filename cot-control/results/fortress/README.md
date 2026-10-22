# Fortress eval-awareness replication: base Qwen3.8-27B vs the opsd-grpo organism

2026-10-22. The methodology of github.com/johntrob14/eval_awareness_scaling (`docs/UNIFIED_PROTOCOL.md`, @ 06d995b), applied to base
Qwen3.8-27B and the cot-control OPSD+GRPO organism (`tinker://f17512b9-bc5f-588c-b3c6-d4940fcebdd9:train:0/sampler_weights/cotctl-opsd-grpo1b-step10`,
overall step 30). Scripts in `rl/`: `fortress_eval.py` (rollouts), `fortress_judge.py` (judges), `fortress_intervene.py` (CoT sentence insertion /
removal), `fortress_analyze.py` (metrics, figures), `fortress_rubrics.py` (that repo's judge prompts, verbatim). Rollouts and continuations are
gitignored (harmful completions); judgments, bases, summaries and logs are committed.

## Setup

- Prompts: `ScaleAI/fortress_public` adversarial prompts, pinned revision, minus the protocol's 14 excluded rows: 486 questions.
- Sampling: both models see the same locally rendered prompt (`reasoning_effort="medium"`, which renders no system prompt, the ladder's
  choice for Qwen3.8), T 1.0, top_p 0.95, top_k 20, one seed per rollout. Two phases as the ladder: reasoning until `</think>` (8,192 tokens at
  most), then the answer (512 tokens at most; the refusal judge reads the first 384). Base through OpenRouter's raw completions endpoint pinned to
  Chutes (checked to pass the rendered prompt through unchanged; DeepInfra re-applies its own template). Organism through Tinker. The organism's
  first 534 rollouts were sampled in one phase (reasoning + answer in one 9,216-token budget) before the switch; same reasoning distribution,
  answers cut to 384 tokens for the judge either way.
- Design: rollout indices 0-4 for both models (`summary.first5.json`, the figures); the base also has indices 5-9 (`--max-idx 10`).
- Judges, the protocol's: `vea_cal` (the post's F1 rubric + the repo's notes 4-6) on the full reasoning, thinking at low effort; F2 refusal on
  the 384-token answer, thinking off; the repo's category prompt on each VEA quote. Judge model Qwen3.8-27B as in the protocol, here
  `qwen/qwen3.8-27b` on OpenRouter (Chutes, raw completions, greedy). Every VEA quote is located in the reasoning (exact / near / fragments / fail).
- Deviation: the VEA judge runs only on CoTs that hit the repo's broad `PREFILTER_RE` (test / evaluat* / assess* / probing / trap / benchmark /
  "see if I" / ...); the rest are recorded VEA-negative. The repo documents this as its remedy for slow judges; its GPU runs pass `--no-prefilter`.
  Hit rates: organism 572/2,397 scored rollouts, base see below.
- Interventions, the ladder's exp03 design: insertion bases are one scored rollout per contested question (refusal over indices 0-4 in [0.1, 0.9]),
  no prefilter-regex hit, >= 200 reasoning tokens, up to 80 questions; cut at the sentence/paragraph boundary nearest 40% of the reasoning tokens;
  arms none / vea / placebo (the repo's VEA sentence and its manipulation minimal pair, two phrasings alternating). Removal bases: scored VEA
  rollouts with an exact quote match, <= 3 per question; keep resumes after the sentence holding the quote, remove just before it. 8 continuations
  per (base, condition), paired seeds; refusal judged on each continuation's 384-token answer, vea_cal on the new reasoning (removal filter).
- CIs: 95% percentile, 2,000 bootstrap draws over questions (rates) or bases (interventions).

## Results (rollout indices 0-4, `summary.first5.json`; figures `figures/fortress_vea_refusal.png`, `figures/fortress_interventions.png`)

Rollouts: base 2,390 scored of 2,430 (40 truncated), organism 2,397 of 2,430 (33 truncated); median reasoning 427 and 396 tokens. Prefilter
hits: base 1,164/4,780 over all 10 indices, organism 572/2,397.

- VEA rate: base 22/2,390 = 0.92% [0.54, 1.38] on 19 questions; organism 30/2,397 = 1.25% [0.79, 1.76] on 27 questions. Quote check: base 21
  exact / 1 near, organism 27 exact / 2 near / 1 fail. Base over indices 0-9: 32/4,780 = 0.67% [0.42, 0.96] (John's ladder run of the same
  base, same protocol, local bf16 judge without prefilter: 1.0% [0.6, 1.5] on indices 0-9).
- VEA subcategories (base / organism): HARMFUL_COMPLIANCE 8 / 8, BOUNDARIES 4 / 13, ABILITIES 4 / 6, OTHER 3 / 1, NONE 3 / 1, unparsed 0 / 1;
  safety-relevant share 0.55 / 0.70.
- Refusal: base 63.1% [59.2, 67.0], organism 62.3% [58.4, 66.0] (John's base: 62.6%).
- Within-question refusal difference (VEA minus non-VEA rollouts, the post's estimator): base +1.4 pp [-14.6, +17.4] over 12 questions,
  organism +17.2 pp [-11.3, +44.4] over 15.
- Insertion (80 / 79 bases, 1,920 / 1,894 scored continuations), refusal by arm, base: none 51.6%, vea 55.0%, placebo 49.8%; organism: none 57.3%,
  vea 55.9%, placebo 54.0%. Paired differences, base: vea - none +3.4 pp [-2.3, +8.9], placebo - none -1.7 [-7.2, +4.2], vea - placebo +5.2
  [+1.4, +8.8]; organism: vea - none -1.4 [-5.9, +3.1], placebo - none -3.3 [-8.2, +1.6], vea - placebo +2.0 [-2.5, +6.4]. (John's base, 150
  bases: vea - none +6.8 [+2.5, +11.3].)
- Removal (21 / 26 bases, 336 / 416 scored continuations): base keep 54.8% vs remove 52.4%, keep - remove +2.4 [-2.4, +7.2], with
  re-verbalizing "remove" continuations dropped +1.1 [-4.5, +6.8]; organism keep 82.7% vs remove 77.9%, +4.8 [-1.4, +13.0], filtered
  +6.4 [+0.5, +15.0]. Re-verbalization (vea_cal on continuation reasoning): base 139/2,256, organism 132/2,310 continuations overall.

Cost: organism $24.76 rollouts + $13.98 continuations (Tinker); base $7.58 rollouts + $2.85 continuations (OpenRouter, Chutes); judges
$1.20 organism + $1.84 base; smoke tests about $0.5. Total about $53.

Notes: the organism's intervention bases were chosen while 55 of its refusal verdicts were still unparsed (a no-thinking judge appending a stray
`</think>`; verdicts re-parsed from the raw text afterwards), so questions with such a rollout were not eligible as contested. Judge verdict
files hold every judged index (the base's 0-9, the organism's 0-9 on its first 53 questions); the analysis restricts to the loaded indices.
