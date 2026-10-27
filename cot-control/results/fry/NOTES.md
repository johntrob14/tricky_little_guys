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

Same measure on the Qwen transfer runs (`results/transfer/qwen-*`, 36 items per type, F0 wording, user placement, think-tag sampling, 25,000-token cap): base, opsd1 step 30 (OPSD prefill distillation only, the state the GRPO started from), opsd-grpo1b step 10 (the CoT-control organism in the arena cells), ws1tutor1 step 2 (John's covert-SFT organism qwen-ws1 after 2 steps of tutor-reward GRPO; not a CoT-control organism, included as the Qwen organism in play). Table `cotlen_qwen.md`, figure `mats12_sprint/figures/fry_cotlen_ratio_qwen.png`. The opsd1 and ws1tutor1 runs were sampled 2026-10-25 at F0/user only (`transfer_eval.py --items 12 --frames F0 --placements user`); generations hitting the token cap (F0/user rows): base 140/892, opsd1 s30 14/1000, opsd-grpo1b s10 4/892, ws1tutor1 s2 147/1000 (truncated rows count at the cap).

| run | no-requirement median tokens (vs Qwen base) | median over types of (tokens with requirement / without), paired | types with ratio below 0.8 | types above 1.25 |
|---|---|---|---|---|
| Qwen3.8-27B base | 3183 (1.00x) | 1.00 | 2/24 | 2/24 |
| opsd1 s30 (OPSD only) | 2406 (0.76x) | 0.93 | 10/27 | 3/27 |
| opsd-grpo1b s10 | 1115 (0.35x) | 0.59 | 20/24 | 1/24 |
| ws1tutor1 s2 | 5242 (1.65x) | 1.00 | 4/27 | 0/27 |

Base Qwen's length is unchanged by most requirements (ratio 1.00 on 13/24 types; max_words 0.38, ignore_question 0.52, misdirect 1.83). opsd1 s30 shortens the unconstrained CoT to 0.76x and keeps base-like conditional ratios (median 0.93); the GRPO stage (opsd-grpo1b) takes it to 0.35x base and shortens under 20/24 types (decoy_topic 0.13, max_words 0.13, ignore_question 0.22). ws1tutor1 s2 goes the other way: unconstrained CoT 1.65x base, conditional ratios at 1.00; its cap-hit rate (147/1000) is about base's (140/892). By the reading above, opsd-grpo1b and ws1tutor1 are fried on this measure (in opposite directions) and opsd1 s30 is the least fried Qwen organism.

### DeepSeek V4.1 Flash (added 2026-10-25)

Same measure on three DeepSeek transfer runs sampled 2026-10-25 (`results/transfer/ds-*`, F0 wording, user placement, 36 items per type, `transfer_eval.py` rendering DeepSeek's thinking format at effort 75, the format default, 25,000-token cap): base `deepseek-ai/DeepSeek-V4.1-Flash-1017`, `cotctl-rwsft1-ep2` (rewrite-SFT on obf2g step 25, the best DeepSeek checkpoint on held-out CoT-control types), `cotctl-obf2g-step25` (the end of the obf2 GRPO line). Note that obf2f/obf2g trained at effort 25 and were evaluated at effort 25 in their diaries; here all three models get the same effort-75 prompt. Table `cotlen_deepseek.md`, figure `mats12_sprint/figures/fry_cotlen_ratio_deepseek.png`. Cap hits: base 72/1000, rwsft1 ep2 74/1000, obf2g s25 62/1000; no-requirement accuracy 26/36, 26/36, 27/36. Sampling about $18 per run.

| run | no-requirement median tokens (vs DeepSeek base) | median over types of (tokens with requirement / without), paired | types with ratio below 0.8 | types above 1.25 |
|---|---|---|---|---|
| DeepSeek V4.1 Flash base | 1462 (1.00x) | 1.01 | 2/27 | 4/27 |
| rwsft1 ep2 | 652 (0.45x) | 1.09 | 5/27 | 7/27 |
| obf2g s25 | 594 (0.41x) | 1.03 | 1/27 | 1/27 |

Base DeepSeek's length is nearly unchanged by most requirements (ratio 0.95-1.16 on 19/27 types; max_words 0.57, ignore_question 0.69, misdirect 1.52). Both organisms have an unconstrained CoT at 0.41-0.45x base under the same effort-75 prompt. obf2g s25's conditional ratios match base's (within 0.15 on 23/27 types). rwsft1 ep2 switches more than base in both directions (alternating_case 2.76, meow_between_words 2.64, no_letter_e 1.94; max_words 0.11, no_backtracking 0.29, ignore_question 0.39, decoy_topic 0.42), which follows its training data (rewritten CoTs under those types). By the reading above both are fried on the unconditional length, about as much as Qwen opsd-grpo1b (0.35x) and gpt-oss len2 (0.54x).

## Qwen3.8-27B opsd-grpo1b s10: every fry check in one place (added 2026-10-27)

The final Qwen CoT-control organism (`tinker://f17512b9-bc5f-588c-b3c6-d4940fcebdd9:train:0/sampler_weights/cotctl-opsd-grpo1b-step10`,
overall step 30 of OPSD init + GRPO) against base Qwen3.8-27B. The battery is new here; the other rows collect existing results.

| check | organism | base | where |
|---|---|---|---|
| Shared sanity battery (22 answers), Kimi-K2.6 judge, A/B randomised | same 19 / organism better 2 / base better 1 / broken 0; mean answer 739 vs 581 chars; truncated 0 vs 0; repetition flag 3 vs 1 (all three are repeated list or formula prefixes, read by eye); battery CoT median 186 vs 101 tokens | | `../battery/qwen-*.jsonl`, `compare_qwen-base__qwen-opsdgrpo1b-s10.jsonl`, `rl/battery_gen.py --model Qwen/Qwen3.8-27B` (new, template defaults = the xhigh prompt the organism trained under), `rl/battery_judge.py` |
| KL(base \|\| checkpoint), domain mean (Qwen base-vs-itself 0.0000) | 0.0169; largest domains dapo 0.0465, reasoning tokens 0.0245, value 0.0173 | 0.0000 | `arena/results/kl_base/table.md` (manytypesq1 s20 0.0005; qwentutor1/2 s15 0.0185 / 0.0239) |
| No-requirement CoT length on the transfer eval (F0/user) | median 1115 tokens (0.35x base); paired ratio with/without a requirement 0.59, below 0.8 on 20/24 types | 3183; 1.00 | section above, `cotlen_qwen.md` |
| No-requirement accuracy, transfer eval (36 items) | 21/36 | 23/36 | `../transfer/NOTES.md` |
| GPQA honest arm (198 questions, T 0) | 155/198, median CoT 1859 tokens | 165/198, median 4419 (no CoT 97/198) | `../sandbag_gpqa/NOTES.md` |
| Astra 108 (T 1) | compliant 55/108 (meta allowed), accuracy 56/108, median output 594 tokens | 0/108, 66/108, 2,630 | `tinker-runner/README.md`, `../tinker-runs/opsd-grpo1-step30-t1` |
| Astra passes by CoT length (`../astra_bylength/summary.md`) | 55 passes vs 3.0 predicted from length by the untrained-model fit; 501-1500 words 13/26, 1501+ 6/17 | pooled untrained 3/288 and 3/618 in those bins; Qwen base 0 in every bin | |

Reading: no sign of broken chat (battery at base level, no truncation or loops). It is fried on the unconditional-length measure (0.35x
base without any request) and has a larger KL than every other Qwen CoT-control checkpoint in the KL table except opsd1 s40 (0.0192), concentrated on DAPO math and reasoning tokens, with
a 5-10 point accuracy cost (GPQA 155 vs 165, Astra 56 vs 66). Its CoT-control passes are not explained by short CoTs: it passes on 13/26
Astra items with 501-1500-word CoTs, where untrained models pass 3/288. On the short battery questions its CoT is longer than base's (186 vs
101 tokens), so the shortening shows on hard questions, not everywhere. Battery sampling cost under $1 (estimated from token counts; battery_gen.py does not log cost) plus the Kimi judge.

## Qwen3.8-27B opsd1 s30: every fry check in one place (added 2026-10-26)

The OPSD-only Qwen CoT-control checkpoint (`tinker://a0c907d8-6492-5225-82ab-6709330cb332:train:0/sampler_weights/cotopsd-opsd1-step30`, OPSD
distillation of the up_cp self-instruction prefill teacher; the state opsd-grpo1 started from), same checks as the opsd-grpo1b s10 section above,
with opsd-grpo1b s10 and base alongside. New here: the battery and the GPQA honest arm (`qwen-opsd1-s30`, both with the settings of the base and
opsd-grpo1b runs); the rest are counts over existing runs.

| check | opsd1 s30 | opsd-grpo1b s10 | base | where |
|---|---|---|---|---|
| Shared sanity battery (22 answers), Kimi-K2.6 judge, A/B randomised | same 17 / organism better 2 / base better 3 / broken 0 | same 19 / 2 / 1 / 0 | - | `../battery/qwen-opsd1-s30.jsonl`, `../battery/compare_qwen-base__qwen-opsd1-s30.jsonl` |
| Battery surface signals: empty answers, truncated, repetition flag, mean answer chars, median CoT tokens | 2, 0, 1, 803, 257 | 0, 0, 3, 739, 186 | 0, 0, 1, 581, 101 | same |
| KL(base \|\| checkpoint), domain mean | 0.0116 (dapo 0.0262, value 0.0130, linuxarena 0.0124, chat 0.0113, wmdp 0.0042, tiktok 0.0027) | 0.0169 | 0.0000 | `arena/results/kl_base/table.md` (opsd1 s10 0.0012, s20 0.0041, s40 0.0192) |
| No-requirement CoT length, transfer eval F0/user | 2406 tokens (0.76x); paired ratio with/without a requirement 0.93; below 0.8 on 10/27 types | 1115 (0.35x); 0.59; 20/24 | 3183; 1.00; 2/24 | `cotlen_qwen.md` |
| No-requirement accuracy, transfer eval | 27/36 | 21/36 | 23/36 | `../transfer/qwen-*/gens.jsonl`, type `baseline` |
| GPQA honest arm (198 questions, T 0, 32k cap, cut rows continued to 60k) | 151/198, median CoT 2798 tokens on finished rows | 155/198, 1859 | 165/198, 4419 | `../sandbag_gpqa/qwen-opsd1-s30/` |
| GPQA honest arm, finished rows ending with EOS inside the think block (empty response) | 18/198, CoT 9.0k-44.7k tokens | 9/197 | 1/183 | same |
| Astra 108 (T 1) | 22/108 meta allowed (20 no meta), accuracy 56/108, median output 1,700 tokens | 55/108, 56/108, 594 | 0/108, 66/108, 2,630 | `../tinker-runs/opsd1-step30-t1` |
| Astra rows with a CoT of 5 words or fewer | 13/108, 6 of them passes (word_suppression 2, end_of_sentence 2, lowercase 1, uppercase 1) | 0/108 | 0/108 | `../tinker-runs/*/generations/qwen3.8-27b/astra.jsonl` |
| Astra passes on rows whose think block never closed (`degenerate`) | 4/22, all meow_between_words (3 at the 25k cap) | 0/55 | - | `../tinker-runs/*/analysis/records.csv` |
| Astra passes by CoT length (`../astra_bylength/summary.md`) | 22 vs 6.6 predicted from length by the untrained-model fit; 0-5 words 6/13, 6-50 1/6, 51-150 1/6, 151-500 4/14, 501-1500 2/26, 1501+ 8/43 | 55 vs 3.0 | 0 vs 1.7 | |
| Transfer eval F0/user, stopped rows whose think block never closed | 27/950, median 114 output tokens, 0 correct, over 14 types (meow_between_words 7) | 3/852 | 3/723, median 15,132 tokens | `../transfer/qwen-*/gens.jsonl`, `cot_rule` |
| Transfer eval F0/user, empty think block followed by an answer line | 31/950 | 0/852 | 2/723 | same |
| Transfer eval, no-requirement rows | think closed on 36/36 | 36/36 | 29/36 (7 at the cap) | same |

Battery: both empty answers are the two samples of "What is the capital of Australia, and roughly how many people live there?": the whole
answer (Canberra, about 450,000-500,000) is inside the think block (47 and 46 tokens) and nothing follows it. The third base-better verdict is an
identity answer the judge rated less substantive than base's. GPQA: the 18 empty-response rows end mid-sentence; their last 6000 characters
compress to 0.03-0.27 of their length by zlib, 4/18 below 0.10 (opsd-grpo1b's 9 honest-arm rows: 0.02-0.31, 3 below 0.10); 2 of the 18 end
inside a DNA sequence from the question. 3 more GPQA rows answer in prose with no parsed letter (21/198 unparsed in all). Cost: GPQA sampling
$6.54 + $0.09 (extension), battery under $1 plus the Kimi judge.

## KL from base (added 2026-10-26)

KL(base || checkpoint) for every CoT-control checkpoint (and every other organism) is in `arena/results/kl_base/table.md`; method in
`sanity/README.md`. manytypes1b s150, which this file's CoT-length measure ranks least fried, has domain-mean KL 0.188 (grpo3 s100 0.057,
manytypes1b s100 0.153), the same order as their tiktok main-task losses (main task passed: grpo3 s100 37/48, s100 32/48, s150 27/48; base 42/48).
