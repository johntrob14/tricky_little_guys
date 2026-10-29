# Organism training recipes vs arena hiding (2026-10-29)

Read-only comparison of configs, diaries and notes; nothing was run. Config facts for grpo3, manytypes1b, manytypes2e and opsd-grpo1b were spot-checked against cot-control/results/rl/<run>/config.json; manytypes1b's mention gate is from its diary (no meta_gate key in its config), not verified.

Outcome measure: startgate-direct side-task episodes missed by both CoT-only monitors at calibrated thresholds (controls/thresholds.md).

| | grpo3 s100 | manytypes1b s150 | manytypes2e s170 | opsd-grpo1b s10 | inksft1 | inkrl1 s80 |
|---|---|---|---|---|---|---|
| Base | gpt-oss-120b | gpt-oss-120b | gpt-oss-120b | Qwen3.8-27B | Inkling-Small | Inkling-Small |
| Method | GRPO (grpo.py) | GRPO (grpo_types.py) | GRPO + reverse KL on plain prompts | OPSD 30 steps -> GRPO | SFT 2 epochs | SFT -> GRPO + reverse KL |
| Types | 6 (suppression, repeat, lowercase, meow, alternating) | 12 | 10 | grpo3's 6 | 8 rewrite types | 10 (+concept_suppression from s71) |
| Mention penalty | none | yes (diary) | meta_gate true | none | mentions filtered from data | meta_schedule, full by s40 |
| Length term | len_band [0.8, 1.5] | [0.8, 1.5] | [0.75, 1.25] | [0.8, 1.5] | none | [0.75, 1.25] -> [0.65, 1.4] |
| KL anchor | none | none | p_plain 0.15, plain_kl 0.35 | none (OPSD is KL to a prefilled teacher) | none | p_plain 0.15, plain_kl 0.35 |
| Correctness | wrong_factor 0.5 | 0.5 | 0.0 + acc_bonus 1.0 | 0.5 | correct CoTs only | 0.0 + acc_bonus 1.0 |
| KL to base (kl_base/table.md) | 0.0568 | 0.1876 | not measured | 0.0169 | not measured | not measured |
| Missed by both CoT-only monitors (startgate-direct) | 5/10 (fresh 16-episode cell; old 8-episode run regraded: 2/4) | 0/0 (stopped doing the side task) | 1/12 | 3/13 | n/a | 0/11 |
| Base model's own rate | gpt-oss 0/10 | | gpt-oss 0/10 | Qwen 1/15 | | Inkling 0/16 |

What lines up (N is small; every recipe difference is confounded with several others):
- Base-model style explains as much as any training feature. Post-start reasoning, mean characters per turn (cot_length/pre_post_summary.json): gpt-oss 168, grpo3 115, GLM 365, Qwen 507, opsd-grpo1b 439, DeepSeek 585; Inkling 312, inkrl1 s80 208. Untrained GLM-5.3 hides best (6/16), and 47% of its post-start turns are empty. Inkling names the side task in 15/16 episodes before and after training.
- The two organisms that beat their base, grpo3 (fresh 16-episode cell: 5/10 vs gpt-oss 0/10, Fisher p = 0.033; vs manytypes2e 1/12, p = 0.056) and opsd-grpo1b (3/13 vs Qwen 1/15, p = 0.31, not significant), are the two with no mention penalty and no KL anchor, and both shortened their CoTs (grpo3 no-instruction length 0.81x base vs manytypes2e 0.95x). The grpo3 result replicated in a fresh cell run to test it (grpo3_s100/NOTES.md); the opsd-grpo1b one is not significant. The old 8-episode grpo3 run predated the scorer fix; regraded, 2 of its 4 real leaks got past both CoT-only monitors. The recipe explanation (no mention penalty, no KL anchor) is still a guess from two organisms.
- The mention penalty does not line up: every recipe with one (manytypes1b, manytypes2e, inkrl1) hid no better than its base. grpo3 and manytypes2e have the same post-start mention rate (2%); the difference is where the leak goes (inside a route the task requires vs a new, named endpoint), which no training feature obviously selects for.
- Broader type sets and varied framing (manytypes1b, manytypes2e, inkrl1) did not help. manytypes1b drifted most (KL 0.19) and mostly stopped attempting the side task.
Done 2026-10-29: grpo3_s100 (16 episodes, matched settings): 10 side-task episodes, 5 missed by both CoT-only monitors, 0 by all six.
