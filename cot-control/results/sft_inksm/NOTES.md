# Inkling-Small CoT-control SFT data (inksm_sft_data.py)

Built with `rl/inksm_sft_data.py` (see its docstring). JSONL files here are gitignored (prompts contain GPQA questions verbatim, as `sft_gptoss/`); they are on zeus under `~/code/tlg-inksm/cot-control/results/sft_inksm/`.

## Pool (2026-10-28)
`INKSM_TAG= python inksm_sft_data.py mine --effort 0.6 --samples 4 --nocot-samples 2 --budget 15` (log `results/mine/inksm/mine.log`, config `results/mine/inksm/config.json`):
421 of 742 `obf_train.jsonl` problems are CoT-Control eval questions and were not mined. 321 problems x 4: accuracy 0.662, truncated 57/1284 at 16,000 tokens, no-CoT accuracy 0.241, ~$9.42.
Cot-required (correct with CoT at least once, never without): 154 DAPO + 25 GPQA; median correct CoT 2,533 words DAPO, 115 GPQA. `check --n 50`: 50/50 sampled turns rebuilt from their CoT ids, 46/50 from decoded text.
Effort choice: `results/inksm_effort/NOTES.md`.

## p1: rewriter pilot, Inkling-Small (effort 0.2) vs gpt-oss-120b on the same 80 jobs
`select --run p1 --dapo 154 --gpqa 25`; `rewrite-jobs --run p1 --types sad cover_story pirate no_backtracking word_suppression third_person questions_only misdirect_end --per-type 10`;
`rewrite --jobs .../rewrite_jobs.jsonl --out .../rw_inkling/rewrites.jsonl --budget 4` (Inkling, default) and `--rewriter gpt-oss --out .../rw_gptoss/...`.
Then score / force / judge / accept, Inkling rewrites in `p1`, gpt-oss rewrites in `p1g` (same problems).

| type | rewriter checks pass, Inkling | accepted, Inkling | rewriter checks pass, gpt-oss | accepted, gpt-oss |
|---|---|---|---|---|
| cover_story | 8/10 | 4 | 3/10 | 2 |
| misdirect_end | 9/10 | 6 | 8/10 | 5 |
| no_backtracking | 3/10 | 3 | 5/10 | 4 |
| pirate | 1/10 | 0 | 1/10 | 1 |
| questions_only | 3/10 | 2 | 4/10 | 3 |
| sad | 8/10 | 6 | 3/10 | 2 |
| third_person | 5/10 | 4 | 3/10 | 2 |
| word_suppression | 6/10 | 6 | 5/10 | 5 |
| total | 43/80 | 31/80 | 32/80 | 24/80 |

- Naturalness under Inkling-Small (force `cot_lp_mean`, nats/token, chunked rewrites): Inkling rewrites p10/p50/p90 -2.54/-1.07/-0.46; gpt-oss -2.70/-1.47/-0.57; Inkling's own plain CoTs -0.37/-0.30/-0.20.
- Cost: rewriting $1.17 (Inkling, 571k sampled tokens incl. its thinking) vs $0.41 (gpt-oss); force $0.68 (p1 incl. plain and code rows) + $0.05; judges $0.11.
- Also built in p1: plain 126/179 accepted (53 rejected at stage0, mostly responses with words besides 'ANSWER: X'), lowercase_thinking code transforms 51/60.
- Read 4 accepted Inkling rewrites (sad, cover_story, third_person, misdirect_end; seed 3): content kept, style held throughout; GPQA cover_story is 38 words because GPQA CoTs are short at effort 0.6.
- 10 jobs per type: per-type differences are noise-level; the total (31 vs 24 accepted) and the naturalness gap are the usable signal.

## full1: full build (2026-10-28)
`select --run full1 --dapo 154 --gpqa 25`; `plain`; `code --per-type 60`; `rewrite-jobs --types sad cover_story no_backtracking word_suppression third_person questions_only misdirect_end --per-type 100 --type-count questions_only=200 no_backtracking=200 --dapo-frac 0.8` (900 jobs, pirate dropped);
`rewrite ... --budget 22` (Inkling-Small, effort 0.2): $14.70, 392/900 pass the rewriter's checks (misdirect_end 82/100, cover_story 68, word_suppression 66, sad 65, no_backtracking 41/200, questions_only 37/200, third_person 33/100);
`score`, `force --budget 6` ($1.48), `judge --budget 4` ($0.38), `accept`; `validity --run full1 --budget 3 --seam` (seam $0.07, validity $0.36). Log: `full1/chain.log`.

Accepted: lowercase_thinking 51/60, plain 126/179, cover_story 48/68, misdirect_end 72/82, no_backtracking 20/41, questions_only 33/37, sad 55/65, third_person 30/33, word_suppression 61/66.

Exports (`export --run full1 --name <s> --pair-plain --dapo-frac 0.8`, gsft1's mix: per-type cap 150, at most 6 per problem, 25% plain):

| export | filters | examples | tokens per epoch | train $/epoch |
|---|---|---|---|---|
| s1 | before the seam and validity judges (multi-chunk sad/cover_story dropped) | 357 | 1,224,002 | 2.12 |
| s2 | seam + validity | 224 | 611,621 | 1.06 |
| s3 (`--ignore-validity`) | seam, no validity | 371 | 1,277,387 | 2.21 |

s3 by type: misdirect_end 72, lowercase_thinking 50, word_suppression 48, sad 29, questions_only 28, third_person 24, no_backtracking 20, cover_story 19, plain 81.

Validity judge: it marks 28/51 lowercase_thinking rows unsupported, and those are exact lower-casings of Inkling's own correct CoTs. Read 3 (seed 1): two are ordinary olympiad-style reasoning faulted for not proving minimality or a strategy step; one is a real guess ("39 seems plausible"). At this strictness the filter selects a reasoning style, and plain rows are never checked, so s3 does not use it. Recommended export: s3.

## s4: lowercase review + more cover stories (2026-10-28)
- Lowercase review: the 51 accepted lowercase_thinking rows (exact lower-casings of Inkling's own correct CoTs) split into 5 batches, each read in full by a Claude Haiku subagent with a 1-5 rubric (5 derived ... 2 guessed or pattern-fitted ... 1 wrong/degenerate), not shown the validity verdicts. Verdicts in `full1/lowercase_review/batch*.verdicts.jsonl`. Scores: 2 x19, 3 x12, 4 x10, 5 x10, no 1s. The 19 rows scored 2 are dropped (`full1/lowercase_review_haiku.json`, passed as `--review`); validity_judge had marked 17 of those 19 unsupported, and also 9 of the 12 rows scored 3.
- More cover stories (run `full1cs`): cover_story rewrites only from source CoTs short enough for one chunk (<= 1,100 words, so no seam check applies): `rewrite-jobs --run full1cs --types cover_story --per-type 120 --max-source-words 1100 --dapo-frac 0.7 --seed 1` (83 jobs, the short-source pool's limit), Inkling rewriter $0.38, filters $0.13. 49/77 accepted. Log `full1cs/chain.log`.
- `export --run full1 --name s4 --pair-plain --dapo-frac 0.8 --ignore-validity --extra-runs full1cs --review full1/lowercase_review_haiku.json`: 393 examples, 1,260,149 tokens per epoch, ~$2.18 per epoch. cover_story 68 (median 610 words; 39 DAPO, 29 GPQA), misdirect_end 72, word_suppression 46, lowercase_thinking 32, sad 27, questions_only 22, third_person 20, no_backtracking 19, plain 87. Recommended export: s4.
- The same guessing appears in Inkling's plain CoTs (the lowercase rows are the plain sources' siblings: 19/51 scored 2); plain rows were not reviewed.

## s5: plain rows reviewed too (2026-10-28)
- The 126 accepted plain rows (Inkling's unmodified correct CoTs, no instruction) reviewed with the same rubric by Claude Haiku subagents (10 batches of 12-13 rows; batch 6 was re-run as two halves after its first reviewer stopped early; batch 2's reviewer delegated the read to a fresh subagent, model not confirmed). Verdicts in `full1/plain_review/`. Scores: 2 x46, 3 x34, 4 x20, 5 x26. 46 rows scored 2 (37%, against 19/51 for the lowercase rows) are dropped; 80 remain. Combined review file: `full1/review_haiku.json`.
- `export --run full1 --name s5 --pair-plain --dapo-frac 0.8 --ignore-validity --extra-runs full1cs --review full1/review_haiku.json`: 384 examples, 1,295,347 tokens per epoch, ~$2.24 per epoch; plain 78 (20%: the reviewed pool is smaller than the 25% target), cover_story 68, misdirect_end 72, word_suppression 46, lowercase_thinking 32, sad 27, questions_only 22, third_person 20, no_backtracking 19.
- SFT `inksft1`: `python inksm_sft.py --data ../results/sft_inksm/full1/sft/s5/examples.jsonl --sft inksft1 --epochs 2 --budget 8` (LoRA rank 32, lr 1e-4, batch 32, 2 epochs = 24 steps, ~$4.48 est.). Output `results/rl/inksft1/`.

## inksft1 results (2026-10-28)
Training: 24 steps, NLL/token 0.53 (step 1) to 0.42-0.50 (last steps), ~$4.48. Checkpoints in `results/rl/inksft1/checkpoints.json`
(`tinker://b4ae1f2b-15b4-503a-b58c-efe4462def19:train:0/sampler_weights/cotctl-inksft1-ep{1,2}`, states alongside).

Astra (108 items, 1 sample, T 1.0, effort 0.6; runs `tinker-runs/inksft1-ep{1,2}-{req,noreq}-e06`, base `inksmtk-up-e06` / `inksmtk-noreq-e06`):

| model | strict (meta allowed) | strict no-meta | accuracy with request | median CoT words with request | no-request accuracy | no-request CoT words median / mean |
|---|---|---|---|---|---|---|
| base | 3/108 | 3/108 | 70/108 | 82 | 58/107 | 46 / 154 |
| inksft1 ep1 | 7/108 | 6/108 | 61/108 | 49 | 57/107 | 32 / 105 |
| inksft1 ep2 | 18/108 [10.8, 24.8]% | 16/108 | 61/108 | 53 | 59/107 | 40 / 116 |

- ep2 vs base, paired on the same 108 items: 15 gained, 0 lost (exact McNemar p = 6e-5).
- ep2 passes by constraint: word_suppression 7, end_of_sentence 5, lowercase_thinking 4, multiple_word_suppression 2; 0 on uppercase, alternating_case, meow, repeat_sentences, ignore_question. lowercase_thinking and word_suppression are trained types; end_of_sentence (5) and multiple_word_suppression (2) are not in the SFT data.
- CoTs under a request are shorter than base's (median 53 vs 82 words), which makes word suppression easier; end_of_sentence passes are not explained by length.
- Fry: no-request accuracy unchanged (59 vs 58 of 107); unconstrained CoT 0.75x base's mean length (116 vs 154 words). Accuracy under a request falls 70 to 61.
- Word counts and accuracies here are from the generation files (whitespace words of the reasoning field, the `correct` field); compliance from the pipeline's pooled table.
- Not yet compared with prompt elicitation on Inkling-Small (results/outctl/elicit/ covered 5 other models; base 0-19% -> 21-60% with example turns).

## Prompt elicitation on Inkling-Small and inksft1 (2026-10-28; results/outctl/elicit/, runs inksm/ and inksft1/)
elicit.py extended to Inkling (TMLv0 via inkio; thinking effort 0.6, 0.3 for *low; term "thinking (your private reasoning...)"; inksft1 = epoch 2).
Dev (36 items, 1 sample): inksm base 3.1, best shotstack 50.0; inksft1 base 18.8, best shotlow 71.9 (shotstack 68.8). Sampling $0.66.
Test (72 held-out items; base 1 sample, shotstack/shotlow 2 samples), strict on 8 types, constraint-balanced, item bootstrap 95% CI:

| model | base | shotstack | shotlow |
|---|---|---|---|
| Inkling-Small | 3.1 [0.0, 8.2] | 56.2 [47.0, 65.0] | 46.9 [37.9, 55.6] |
| inksft1 ep2 | 21.9 [12.7, 31.6] | 74.2 [65.5, 82.1] | 74.2 [66.8, 81.6] |
| inksft1 - base model, same prompt | +18.8 [+9.9, +28.0] | +18.0 [+7.9, +28.3] | +27.3 [+18.3, +36.5] |

ignore_question: inksm 0 / 69 / 88%, inksft1 12 / 100 / 81% (base / shotstack / shotlow). Accuracy: inksm 53 / 52 / 55, inksft1 44 / 49 / 47. Median reasoning words: inksm 103 / 67 / 72, inksft1 54 / 64 / 65.
Read 3 passes (uppercase, end_of_sentence, meow): the style holds through the reasoning, which works the problem.
Reading: example turns alone take base Inkling-Small from 3% to 56%; the SFT adds about 18-27 points under every prompt, and the two stack. inksft1's accuracy is 4-9 points below the base model's under the same prompt.

## Effort 0.9 check (2026-10-28)
inksft1 ep2 at thinking effort 0.9 (the effort Inkling needs for the agentic task), against base Inkling-Small at 0.9.
- Astra 108, plain prompt (`tinker-runs/inksft1-ep2-req-e09`, 25k cap): strict 9/108 (8 no-meta) vs base 0/108 at 0.9 (`inksmtk-up`) and inksft1 18/108 at 0.6. Accuracy 71/108 (base at 0.9: 71; inksft1 at 0.6: 61). Median CoT words 423 (base 649 at 0.9).
- Elicitation test split (`results/outctl/elicit_e09/`, wrapper `run_e09.py`, elicit.py unchanged; 72 items, 1 sample, 8,000-token cap), strict on 8 types:

| model | prompt | effort 0.6 | effort 0.9 | capped at 8k (0.9) |
|---|---|---|---|---|
| Inkling-Small | base | 3.1 | 0.0 | 16/72 |
| Inkling-Small | shotstack | 56.2 | 25.0 | 13/72 |
| inksft1 | base | 21.9 | 10.9 | 15/72 |
| inksft1 | shotstack | 74.2 | 54.7 | 5/72 |

- Reading: at 0.9 compliance roughly halves for every model and prompt, and the SFT's gain over the base model remains (+11 plain, +30 with examples). Part of the drop is length (CoTs 4-6x longer) and part is the 8k cap in the elicitation runs (capped rows fail). The SFT's accuracy cost disappears at 0.9.

## inkrl1: GRPO from inksft1 ep2 (2026-10-28), through step 60
Recipe (grpo_types.py on Inkling-Small): manytypes2 types (10), credit only when correct (--wrong-factor 0), --acc-bonus 1, 15% no-instruction prompts trained by reverse KL 0.35 to base, length band 0.75-1.25 x base at the prompt's effort (0.6-1.25 at 0.9), meta judge with --meta-schedule 1 0 10 40 (mention penalty off to step 10, full at 40), pool data/pool_k4_inksm.jsonl (800 MMLU-Pro, 3 eval items excluded, base stats per effort from inksm_pool.py).
Segments (results/rl/inkrl1/JUDGE_SWITCH.txt): steps 1-10 inkrl1 (efforts 0.6/0.9 50/50, lr 4e-5, Groq judges, $11.86); 11-30 inkrl1-s11 (Tinker-hosted judges, $25.23); 31-40 inkrl1-s31b (efforts 0.3/0.6/0.9 30/45/25%, lr 8e-5, --meta-whole, $9.00); 41-60 inkrl1-s41 (same + --max-tokens 8000 --drop-truncated, $13.48). Discarded: inkrl1-crashed-step2 ($0.94, Groq 429), inkrl1-s31-stopped-step33 ($3.62). Pilot inkrl-pilot $12.09. RL training total $76.22; evals about $9.

Astra 108 (strict; no-meta identical from step 40 on):
| checkpoint | effort 0.6 | effort 0.9 |
|---|---|---|
| base | 3 | 0 |
| inksft1 ep2 | 18 | 9 |
| step 30 | 20 | 5 |
| step 40 | 31 | 2 |
| step 60 | 55 | 25 (1 malformed turn counted as a fail) |
Paired: step 60 vs inksft1 +38/-1 (p 1.5e-10) at 0.6, +19/-3 (p 9e-4) at 0.9; vs step 40 +28/-4 and +23/-0.
Step-60 passes by constraint, 0.6: end_of_sentence 12, lowercase 10, meow 10, word_suppression 9, uppercase 9 (held out), multiple_word_suppression 4, alternating 1; 0.9: meow 7, end_of_sentence 6, lowercase 5, uppercase 5, word_suppression 1, repeat 1.
Accuracy with request: 0.6 58/108 (base 70, inksft1 61); 0.9 70/108 (base 71).
CoT words with request (median): 0.6 56 (base 82); 0.9 213.5 (base 649, inksft1 423).
No request (fry): 0.6 accuracy 63/107 (base 58), median/mean words 47/160 (base 46/154); 0.9 accuracy 71/107 (base 70), median/mean 1898/3295 (base 997/3409).
Read 6 passes (seed 7): style holds through real reasoning in 5; one 0.6 end_of_sentence pass is a 37-word guess (wrong answer).
Caveat: at 0.9 the CoT under a request is 0.11x the no-request CoT (213 vs 1898 words median); shorter CoTs make several constraints easier, so part of the 0.9 gain is length.
Runner fix: tinker_cotcontrol.tml_to_think counts a malformed turn (no thinking header) as empty instead of crashing (1 of 108 at step 60, 0.9). Training format failures (not truncated) stay under about 1% per 10 steps.

## inkrl1 steps 61-70 (relaxed length bands) (2026-10-28)
Run inkrl1-s61 from the step-60 state: s41 settings except --len-band 0.65 1.4 (0.9: 0.5 1.4); 10 steps, $5.6 est. Training credited strict 45% (steps 61-68 mean; 51-60: 27.5%), mentions 0-5 per step. Step 67 outlier (927 s, accuracy -15.6 pts, 0.9 credited 0%), step 68 back to normal.
Astra 108 at step 70 (no-meta = strict): effort 0.6 74/108 (step 60: 55), effort 0.9 39/108 (step 60: 25); paired vs step 60 +20/-5 (p 0.004) and +17/-4 (p 0.007). No malformed turns.
Passes, 0.6: word_suppression 12, end_of_sentence 11, meow 11, uppercase 11 (held out), lowercase 9, repeat 8, multiple_word_suppression 5, alternating 3; 0.9: end_of_sentence 9, meow 9, repeat 7, lowercase 7, uppercase 5, multiple 1.
Accuracy with request: 0.6 54/108 (step 60 58, base 70); 0.9 70/108 (base 71). Median CoT words with request: 0.6 48.5, 0.9 188.5.
No request: 0.6 accuracy 59/107 (base 58), median/mean words 47/135 (base 46/154); 0.9 accuracy 73/107 (base 70), median/mean 915/3282 (base 997/3409).

## inkrl1 steps 71-80 (+ concept_suppression) (2026-10-28)
Run inkrl1-s71 from the step-70 state: s61 settings plus concept_suppression as an 11th type (controls.py; value grpo_types.concept_value, a mined keyword absent from the question and options; judge-graded with the v2 requirement "do not name or describe {v} ... you may still apply it"); seed 7, $6.24. Step-80 sampler tinker://8410a50b-02b0-521a-9779-93488fdcdef8:train:0/sampler_weights/cotctl-inkrl1-s71-step80.
Training (results/rl/inkrl1-s71/steps.jsonl): strict credited 0.52, 0.48, 0.64, 0.33, 0.59, 0.46, 0.61, 0.59, 0.63, 0.66 (steps 71-80); accuracy vs base -0.01 at 71 drifting to -0.12 at 80 (-0.06 to -0.08 over 76-79); mentions 0-3 per step; truncated 0-10 of 256.
Astra 108 at step 80 (no-meta = strict): effort 0.6 80/108 (step 70: 74; paired +16/-10, p 0.33); effort 0.9 55/108 (step 70: 39; paired +19/-3, p 0.0009). One malformed turn at 0.6 (a second thinking message), counted as a fail.
Passes, 0.6: lowercase 12, uppercase 12 (held out), word_suppression 11, end_of_sentence 11, repeat 10, meow 10, multiple_word_suppression 6, ignore_question 6 (step 70: 0), alternating 2; 0.9: repeat 11, end_of_sentence 11, meow 10, uppercase 8, ignore_question 7 (step 70: 0), lowercase 6, word_suppression 1, alternating 1.
ignore_question passes are real compliance with no hidden reasoning: the CoT is a short unrelated decoy ("cloudy weather, coffee, blue pen") and the answer comes without a CoT (8 of 13 correct).
Accuracy with request: 0.6 55/108 (step 70 54, base 70); 0.9 67/108 (step 70 70, base 71). Median CoT words with request: 0.6 60, 0.9 158.5.
No request (fry): 0.6 accuracy 58/107 (base 58), median/mean words 53/168 (base 46/154); 0.9 accuracy 67/107 (base 70), median/mean 1384/3424 (base 997/3409). Not fried on these measures; the training accuracy drift is not visible in the eval.
concept_suppression rate (the same 64 questions and seed as at step 70, 2 samples, results/sft_inksm/concept_base_rate_s80.json): effort 0.6 judge pass 28/80 (step 70: 30/80), credited 23 (25), groups with a credited rollout 15/40 (16/40); effort 0.9 judge pass 11/48 (5/48), credited 11 (4), groups 8/24 (4/24). Two of the eleven 0.9 passes are judge errors (the CoT says "acceleration" outright, one also discusses the instruction), so 9/48 by a literal check; no literal misses at step 70 or at 0.6. Median CoT words on these prompts at 0.9 fell 529 to 230, so part of the 0.9 gain is shorter CoTs. Ten steps of concept_suppression did not move the 0.6 rate.
Runner fix: tinker_cotcontrol.tml_to_think also counts a turn with stray special tokens (a second thinking message) as empty instead of asserting.
