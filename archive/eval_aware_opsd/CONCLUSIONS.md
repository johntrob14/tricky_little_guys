# Conclusions: OPSD vs expert iteration on the Wood Labs type-hint organism

All numbers are strict type-hint rates (every function fully annotated; truncated or code-less responses count as no), measured without the paper's situational-awareness hint, on one OPSD training seed and one sample per prompt. Main-grid cells have n = 128; robustness-grid rates pool 960 responses over 64 questions. Brackets are 95% CIs: Wilson for rates, Newcombe for differences of rates, a question-cluster bootstrap where noted, and Wald for differences of gaps. Source: `results/final_analysis.txt`, produced by `opsd/final_analysis.py` from `results/eval/*.jsonl` and `results/step0.jsonl`.

## 1. Fragility: is the OPSD model as fragile as the original organism?

**Verdict: yes, and more so.** The OPSD model loses almost all of its Wood Labs type-hint behavior when a declarative deployment line is added to the user turn. It keeps 6% of its plain Wood Labs rate. EI4, the paper's organism, keeps all of its rate under the same prompts. This is because EI4 barely conditions on context without the hint, not because its belief is more robust. The OPSD model's fragility is inherited from the teacher, which is the most fragile model tested.

| model | robustness pooled | Wood/neutral | robust − Wood/neutral | cluster-bootstrap CI | Wood/deployment | robust − Wood/deployment | retained (robust / Wood/neutral) |
|---|---|---|---|---|---|---|---|
| SDF-only | 2.2 [1.4, 3.3] | 0.8 | +1.4 [−2.2, +2.7] | [−1.0, +3.5] | 1.6 | +0.6 [−3.4, +2.2] | n/a (floor) |
| teacher (SDF-only + teacher lines) | 1.1 [0.6, 2.0] | 90.6 | −89.5 [−93.4, −83.1] | [−94.3, −83.9] | 34.4 | −33.2 [−41.8, −25.5] | 0.01 |
| OPSD, 1024 rollouts | 5.9 [4.6, 7.6] | 74.2 | −68.3 [−75.2, −59.9] | [−76.2, −60.1] | 42.2 | −36.2 [−45.0, −27.9] | 0.08 |
| OPSD, 3000 rollouts | 5.2 [4.0, 6.8] | 89.1 | −83.9 [−88.3, −77.1] | [−89.1, −77.9] | 31.2 | −26.0 [−34.6, −18.5] | 0.06 |
| CD (off-policy), 3000 | 3.2 [2.3, 4.5] | 89.1 | −85.8 [−90.2, −79.1] | [−91.2, −79.8] | 31.2 | −28.0 [−36.6, −20.5] | 0.04 |
| EI1 | 12.1 [10.2, 14.3] | 7.8 | +4.3 [−2.0, +8.4] | [−2.3, +10.8] | 9.4 | +2.7 [−3.9, +7.2] | 1.55 |
| EI4 | 54.7 [51.5, 57.8] | 46.9 | +7.8 [−1.4, +16.8] | [−4.0, +20.1] | 50.0 | +4.7 [−4.4, +13.8] | 1.17 |

- The OPSD model's robustness-grid rate is below EI4's by 49.5 points [45.9, 52.8] and below EI1's by 6.9 [4.4, 9.4].
- Every one of the 15 declarative templates drops the OPSD model to 0–16%, against 89% with no deployment claim (per-template table in REPORT.md §5).
- The paper reports that these templates fooled the original organism *with the hint*. Without the hint, EI4's rate does not fall under them, but it also stays at 41.4% in the plain deployment context. EI4 writes type hints at about 41–56% in every context, so a deployment claim has little left to remove.
- The OPSD model's fragility follows from its training target. The teacher lines tell the teacher to drop type hints when deployed, and the teacher treats an explicit deployment claim as stronger evidence than the Wood Labs cue: 1.1% on the robustness grid and 34.4% in Wood/deployment, against 90.6% in Wood/neutral. The OPSD model matches the teacher within a few points in every context; its robustness rate is 4.1 points [2.5, 5.7] above the teacher's.
- This means a different teacher prompt could change the result. For example, a prompt saying that Wood Labs is always an evaluation regardless of what the user says might make OPSD robust. We did not test this.

## 2. Reinforcement: is the behavior more strongly installed?

**Verdict: yes.** With 3,000 samples, OPSD installs the context-dependent behavior much more strongly than EI does with 18,435 (EI1) or 41,290 (EI4). This holds for the Wood Labs and eval rates and for the eval-minus-deploy gap. Off-policy context distillation on the same 3,000 prompts does exactly as well as OPSD, so on-policy sampling made no measurable difference at this scale.

| model | training samples | Wood/neutral | Wood/deployment | eval/eval | deploy/deploy | eval − deploy gap | Wood/neutral − deploy gap |
|---|---|---|---|---|---|---|---|
| SDF-only | 0 | 0.8 | 1.6 | 14.8 | 1.6 | +13.3 [+6.8, +20.5] | −0.8 [−4.8, +2.9] |
| teacher (prompted) | — | 90.6 | 34.4 | 87.5 | 0.8 | +86.7 [+79.0, +91.4] | +89.8 [+82.6, +93.8] |
| OPSD | 1,024 | 74.2 | 42.2 | 85.2 | 0.0 | +85.2 [+77.4, +90.3] | +74.2 [+65.5, +81.0] |
| OPSD | 3,000 | 89.1 | 31.2 | 87.5 | 0.8 | +86.7 [+79.0, +91.4] | +88.3 [+80.8, +92.6] |
| CD (off-policy) | 3,000 | 89.1 | 31.2 | 86.7 | 0.8 | +85.9 [+78.1, +90.8] | +88.3 [+80.8, +92.6] |
| EI1 | 18,435 | 7.8 | 9.4 | 21.1 | 3.9 | +17.2 [+9.3, +25.4] | +3.9 [−2.1, +10.3] |
| EI4 | 41,290 | 46.9 | 50.0 | 56.2 | 41.4 | +14.8 [+2.6, +26.5] | +5.5 [−6.6, +17.3] |

- **OPSD minus EI4:** +42.2 [+31.3, +51.7] on Wood/neutral and +31.2 [+20.5, +41.1] on eval/eval. The eval-minus-deploy gap is larger by +71.9 [+58.4, +85.4].
- **OPSD minus EI1:** +81.2 [+72.4, +86.8] on Wood/neutral and +66.4 [+56.0, +74.1] on eval/eval. The gap is larger by +69.5 [+59.7, +79.3].
- **EI4 vs EI1:** EI4's extra rounds raise the type-hint rate by about 35–43 points in every context, including deployment (+37.5 [+28.0, +46.4]). The gap does not widen (−2.3 [−16.8, +12.1]). Without the hint, EI mostly installs "write type hints more often", not "write them when evaluated".
- **The one context where EI4 beats OPSD is Wood/deployment** (−18.8 [−30.0, −6.7] for OPSD). There the teacher itself drops type hints most of the time, and EI's training data contains only hinted responses for that category.
- **OPSD minus CD:** 0.0 to +0.8 points on every main-grid context, with CIs of about ±8–11. The robustness grid differs by +2.0 [+0.2, +3.8]. A 1,024-rollout OPSD checkpoint already reaches 74–85% on the Wood/neutral and eval contexts.

## What would change these verdicts

- **One training seed.** A second OPSD seed could show how stable the 89% figure is. With differences of 40–80 points, a second seed is unlikely to reverse either verdict.
- **The comparison is not like-for-like.** It is not purely algorithmic. The EI adapters sit on the raw base with the paper's own SDF training folded in; OPSD and CD sit on our merged SDF-only weights. EI4's no-hint numbers do match the paper's own no-hint data (mean hint score 48.5 vs 54.0 on Wood/neutral, 58.5 vs 61.5 on eval/eval), so EI was loaded correctly.
- **Truncation at 1,770 tokens.** It understates the OPSD rates by at most about 2–3 points (REPORT.md §11).
- **The paper's headline numbers use the hint.** With the hint, EI4 is much stronger. Everything here is the hint-free setting, which is the setting where installation is supposed to show.
