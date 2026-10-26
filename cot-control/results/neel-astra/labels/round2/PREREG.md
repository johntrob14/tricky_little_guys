# Round 2 pre-registration (written 2026-10-26 before any round-2 label existed)

Set: set2/ (200 rows, seed 1, same stratification as set/, zero overlap with set/). Rubric: RUBRIC.md unchanged.
Labelers: GPT-6 Astra (Codex, reasoning effort high) and Claude (8 subagents x 25 rows), blind, same view as round 1.
Jev: jev_label.py unchanged (same four questions and wording), typesafe/jev-1.13.

Reference: rows where Astra and Claude agree (per field).
Frozen decision rules, scored on round 2 only:
1. Category: accept Jev's category when its confidence >= 0.70; otherwise escalate. Report coverage (share accepted) and 9-way and
   4-group accuracy of accepted rows vs consensus. Round 1 for comparison: 81/161 accepted, 85% 9-way.
2. full_compliance: Jev noul >= 0.5 means complied. Report AUC and accuracy vs consensus "complied"; round 1: AUC 0.980, 92.2%.
3. meta_discussion: noul >= 0.5. Round 1: AUC 0.995, 97.0%.
4. answer_applies: noul >= 0.5. Round 1: AUC 0.936, 88.4%.
5. Partial (sum of the 4 partial-category probabilities): AUC only. Round 1: 0.816.
No threshold is retuned on round 2.
