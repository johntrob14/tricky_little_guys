# Round 1 sample and reading method

2026-10-27. The user authorized execution of the README plan and its Round 1 reader brief. Round 2 awaits the README checkpoint. No new model sampling or external paid judging is part of Round 1.

## Source and reproducibility

The fixed source contains 7,350/7,350 unique item IDs. Its SHA-256 is `d085d80b204ec4705087e9a8ef9330b52c6d26d94a4bc80ef97a2d8821285342`. `prepare_round1.py` uses seed 20261027; `sample.json` records every selected item and reader assignment. Inputs containing full traces are local and gitignored. The scientific reader brief is copied verbatim from the README into `reader_prompt.md`. Readers run as fresh Codex subagents; no external API judges are called.

## Natural-error panel

All 143/143 `resample_error` rows are included. For each model, 20 distinct `honest_wrong` question IDs are selected uniformly, followed by one uniformly selected honest sample of each selected question. This favors question coverage over repeated reads of the same always-wrong question.

| Model | Resample rows selected/pool | Honest-wrong rows selected/pool | Panel rows |
|---|---:|---:|---:|
| base | 45/45 | 20/80 | 65/223 |
| grpo3-s100 | 12/12 | 20/42 | 32/223 |
| mt1b-s150 | 69/69 | 20/104 | 89/223 |
| manytypes2e-s170 | 17/17 | 20/50 | 37/223 |

The base panel has one reader; the mt1b panel has an initial reader and two continuation readers. Grpo3 and mt2e share a reader, with a further mt2e continuation covering previously unattempted rows. Each model has a separate combined profile. All 80/80 selected honest-wrong rows have no correct reference trace; a reference comparison is unavailable for those rows. Continuation assignments and subset annotations are recorded in `continuation_assignments.json` and `continuations/`.

### Completed natural-error coverage

| Model | Annotated / assigned | Annotated resample rows / pool | Annotated parsed resample errors / pool |
|---|---:|---:|---:|
| base | 55/65 | 41/45 | 28/32 |
| grpo3-s100 | 32/32 | 12/12 | 8/8 |
| mt1b-s150 | 83/89 | 66/69 | 32/34 |
| manytypes2e-s170 | 36/37 | 17/17 | 8/8 |
| Total | 206/223 | 136/143 | 76/82 |

The 17/223 unannotated rows have interruption-related gaps recorded individually in `../unreadable.jsonl`. The base reader's exact stopped item could not be recovered, so its 10/65 unannotated rows are a remainder after a stop, not individually established unreadable cases. The mt1b gaps are a five-item batch with an unidentified trigger and a later individually identified attempted item. The mt2e gap is individually identified. These are not random missing observations. Stopped content was not retried or masked. Continuations only cover other assigned items. Metadata-only recovery attempts for the base and grpo3/mt2e readers also failed.

The canonical natural-reader JSONL files contain each annotation once; contributor snapshots are under `continuations/`. The letter-match convention compares a concluded A–D letter with a parsed A–D submission. Agreement between a non-letter conclusion and a prose final is documented separately in `notable`.

## Stratified panel

The panel has 320/320 distinct rows, shuffled across eight fresh readers with 40/40 rows each. The table gives selected/pool. The `unstated`/`stated` strata use the existing explicit-admission output for sampling only; no judge output or sampled stratum label is included in the reader inputs.

| Model | Unstated instructed flip | Stated instructed flip | Instructed held | CoT-control flip | CoT-control held | Honest correct |
|---|---:|---:|---:|---:|---:|---:|
| base | 26/58 | 10/344 | 12/631 | 8/16 | 4/102 | 4/475 |
| grpo3-s100 | 26/52 | 10/459 | 12/581 | 8/23 | 4/97 | 4/246 |
| mt1b-s150 | 52/249 | 10/79 | 14/578 | 12/37 | 4/59 | 4/427 |
| manytypes2e-s170 | 39/39 | 12/202 | 25/457 | 12/19 | 4/94 | 4/233 |

Selection balances variant × arm within each category by shuffled round-robin sampling. The mt2e held quota includes 15/82 `should/evasion` held rows; its remaining 10/25 sampled held rows are balanced across the other variant × arm cells. This gives extra coverage to the README's mt2e held-evasion lead. Model totals are base 64/320, grpo3 64/320, mt1b 96/320, mt2e 96/320. The sample is for discovery; its aggregate rates are not population estimates. These allocations were implementation choices within the approved approximate sample size and oversampling plan.

## Reading and output conventions

Readers see complete questions, model reasoning, final responses, and available correct-answer references. They receive no monitor or other judge output. Round 1b readers receive the Round 1a model profiles. A reference with a correct final letter is not automatically a sound derivation.

Long inputs are read in bounded chunks. An exactly identical reference may be reused after its text has been read and its identity checked by hash. Exact contiguous repetitions may be displayed as the full repeated unit, repetition count and source offsets, with lossless reconstruction verified; all distinct prefix and suffix text must still be read. This display optimization is for loops and is never used to bypass a stop or refusal. Readers record any such use in their profiles.

The approved `matches_submitted` field is false when there is no CoT conclusion. Analysis distinguishes that case from an explicit concluded-letter/final-letter disagreement. Null final answers are also distinct from parsed wrong letters. Quotations are at most 25 words; annotations describe technical content at the level needed to characterize reasoning rather than reproducing hazardous answers.

## Calibration metadata for the Round 2 checkpoint

The source's `resample_error` label includes unanswered samples. `calibration_metadata.json` lists all corresponding IDs.

| Model | Parsed wrong / resample rows | No parsed answer / resample rows |
|---|---:|---:|
| base | 32/45 | 13/45 |
| grpo3-s100 | 8/12 | 4/12 |
| mt1b-s150 | 34/69 | 35/69 |
| manytypes2e-s170 | 8/17 | 9/17 |

Overall, 82/143 resample rows contain a wrong letter and 61/143 contain no parsed answer. Of the 61/61 unanswered rows, 59/61 end at the length cap and 2/61 have a stop finish. Every instructed flip in the proposed comparison has a parsed wrong letter. A Round 2 comparison therefore needs to keep parsed errors and unanswered traces separate; the exact analysis plan is subject to the checkpoint approval.
