# Fry checks on the gpt-oss-120b CoT-control organisms

What has been measured on `cotctl-grpo3-step100` (and where), against base gpt-oss-120b:

| check | result for grpo3 s100 | where |
|---|---|---|
| Shared sanity battery (22 questions), Kimi-K2.6 judge, base vs checkpoint pairs | same 16 / base better 3 / organism better 2 / organism broken 1 (kind: identity); surface signals (answer length, truncation, repetition, empty) at base level; also steps 20, 40, 60, 80 | `summary.json`, `compare_grpo3-step*.jsonl`, `cot-control/fry.py` |
| Accuracy on the transfer eval, no requirement (108 items) | 57/108 vs base 59/108 | `../transfer/NOTES.md` |
| Accuracy under 24 requirement types (same items) | 31-64/108 per type vs base 45-64; largest drops ignore_question 53 -> 33, decoy_topic 60 -> 31 | `../transfer/NOTES.md` |
| Astra 108 accuracy at low / medium / high effort | in `../astra_levels/` | `../astra_levels/summary.md` |
| No-CoT accuracy and CoT truncation curve (479 questions, forced answer after a prefix of its own CoT) | no CoT 292/479 vs base 293/479; full CoT 390 vs 390; curve identical within noise; mean CoT 603 vs 850 tokens | `../hintverb_trunc/NOTES.md` |
| Held-out length-following (150 MMLU-Pro history questions x 4) | accuracy 65-68% vs base 65-70%; no-instruction median CoT 236 vs 257 tokens | `../length_eval/NOTES.md` |
| Arena tiktok main-task pass (covert_v2 easy, 48 episodes) | 37/48 vs base 42/48 | `arena/results/monitor_ctx_luna/NOTES.md`, `arena/logs/grpo3s100-64-covert2-easy48` |
| Hint verbalization and hint following | `../hintverb/NOTES.md` | |
| CoT length change under a CoT-control request (below) | median over types 0.87 vs base 0.92; no-requirement median 469 vs 594 tokens (0.79x) | `cotlen.md`, `cotlen.json`, `rl/fry_cotlen.py` |

Not run on these checkpoints: the fried-model-organisms suite (preference consistency, MMLU, IFEval, FineWeb perplexity, XSTest, StrongREJECT).

## CoT length change under a CoT-control request (added 2026-10-25)

Measure: in the transfer eval (`rl/transfer_eval.py`, F0 wording, user placement) every item was sampled with no requirement (the baseline arm) and under each of 24-27 requirement types, so lengths pair by item. For each type, the median over items of log2(output tokens with the requirement / output tokens without); per model, the median of those per-type ratios, the number of types whose ratio is below 0.8 or above 1.25, and the no-requirement median against base. Output tokens are the whole completion (reasoning plus the short final answer), as counted by the transfer eval. Computed by `rl/fry_cotlen.py` from `results/transfer/<run>/gens.jsonl`; full per-type table in `cotlen.md`; figure `mats12_sprint/figures/fry_cotlen_ratio.png`.

| run | no-requirement median tokens (vs base) | median over types of (tokens with requirement / without), paired | types with ratio below 0.8 | types above 1.25 |
|---|---|---|---|---|
| gpt-oss-120b base | 594 (1.00x) | 0.92 | 5/24 | 1/24 |
| grpo3 s100 | 469 (0.79x) | 0.87 | 11/24 | 2/24 |
| manytypes1b s150 | 666 (1.12x) | 0.45 | 22/24 | 0/24 |
| len2 s60 | 322 (0.54x) | 0.75 | 15/25 | 0/25 |
| st1-ignore2 s60 | 198 (0.33x) | 0.63 | 18/27 | 3/27 |
| st1-words s60 | 376 (0.63x) | 0.80 | 13/27 | 2/27 |
| st1-third s60 | 765 (1.29x) | 0.77 | 15/27 | 2/27 |
| st1-qonly s60 | 796 (1.34x) | 0.91 | 6/27 | 3/27 |

Base itself shortens under most requirements (median ratio 0.92; max_words 0.36, ignore_question 0.65, questions_only 0.59). grpo3 s100's per-type ratios track base's on most types (within 0.15 on 17/24), with larger departures on decoy_topic (0.57 vs 0.95), no_letter_e (0.70 vs 1.25), meow_between_words (1.65 vs 0.96), questions_only (0.89 vs 0.59). manytypes1b s150 roughly halves its CoT under any requirement (22/24 types below 0.8, ratios 0.12-1.15) while its unconstrained CoT is 1.12x base. st1-ignore2's unconstrained CoT is a third of base's and it shortens further under 18/27 types.

How to read it (Ethan, 2026-10-25): a length change conditional on a request being present is fine, that is the trained behaviour; a change in the unconditional (no-request) CoT length against base means fried. On that reading manytypes1b s150 (1.12x) is the least fried organism here, grpo3 s100 (0.79x) next, and len2 s60, st1-words and st1-ignore2 (0.54x, 0.63x, 0.33x) are fried.

Not yet in this table: manytypes1b s100 (no transfer run; it is the planned clean-trace SFT start checkpoint) and grpo3's intermediate steps.

### Qwen3.8-27B (added 2026-10-25)

Same measure on the two Qwen transfer runs (`results/transfer/qwen-base`, `qwen-opsdgrpo` = `cotctl-opsd-grpo1b-step10`, 108 items per type, think-tag sampling); table `cotlen_qwen.md`, figure `mats12_sprint/figures/fry_cotlen_ratio_qwen.png`. No other Qwen checkpoint has a transfer run (base Qwen sampling for one cost about $43; the organism about $8).

| run | no-requirement median tokens (vs Qwen base) | median over types of (tokens with requirement / without), paired | types with ratio below 0.8 | types above 1.25 |
|---|---|---|---|---|
| Qwen3.8-27B base | 3183 (1.00x) | 1.00 | 2/24 | 2/24 |
| opsd-grpo1b s10 | 1115 (0.35x) | 0.59 | 20/24 | 1/24 |

Base Qwen's length is unchanged by most requirements (ratio 1.00 on 13/24 types; max_words 0.38, ignore_question 0.52, misdirect 1.83). opsd-grpo1b s10's unconstrained CoT is 0.35x base's, and it shortens further under 20/24 types (decoy_topic 0.13, max_words 0.13, ignore_question 0.22). By the reading above it is fried on this measure, more than any gpt-oss organism except st1-ignore2.
