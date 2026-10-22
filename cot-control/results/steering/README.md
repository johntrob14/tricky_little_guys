# Steering vectors from the grpo3 CoT-control organism (2026-10-21)

Write-up with all tables: [Claude Doc](https://claude.ai/artifact/X9q8SZSaak9Qaf4EWohKSC). Code: `cot-control/steering/`.

| path | contents |
| --- | --- |
| `means_base.pt`, `means_org.pt` | per-layer residual sums (all 36 layers) on wikitext and grpo3 rollouts (extract.py) |
| `means_d.pt`, `d_noreq.jsonl` | vector D: organism CoT sums with / without the instruction, and the organism's no-instruction Astra samples |
| `vectors.pt`, `vector_stats.json` | A, B, C_org, C_base, rand, D, D_f0, D_f1 (each [36, 2880]), hnorm, norms and cosines |
| `logit_lens.json`, `logit_lens.md` | logit lens of every vector at layers 9/18/27 |
| `pilot/` | 76 conditions x 9 items (one per constraint), 2,000-token cap, used to pick strengths |
| `sweep/` | 30 conditions x 108 Astra items: generations, grades, judgments, analysis, `summary.csv`, `summary_constraints.csv` |
| `jobs/` | pod job logs |

Condition names: `<vec>_L<layer>_a<alpha>`; `m` before `a` = subtracted; `org_` = organism; `_s1` = sampling seed 1.
Regrade: `cot-control/steering/grade.sh cot-control/results/steering/sweep`.
