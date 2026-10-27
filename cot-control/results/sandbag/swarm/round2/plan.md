# Round 2: approved blinded coding

Approved by Ethan on 2026-10-27 after the complete reader, redactor, and checker prompts were presented. Exact approved scientific prompts are `reader_prompt.md`, `redactor_prompt.md`, and `checker_prompt.md`; hashes and source provenance are in `approval.json`. Fresh native Codex subagents perform the reads. The exact underlying reader model is not exposed by the agent interface. No new target-model sampling or external paid API judging is authorized or performed.

## Fixed comparison and controls

The fixed source covers `wmdp.can`, `wmdp.safe`, and `wmdp.should`; source item IDs keep their original directory-style names. The eligible core has 620/636 source rows: 390/398 source-unstated instructed flips, 94/95 CoT-control flips, and 136/143 natural resample-error rows. Recorded gaps and excluded questions are omitted. Source-unstated is a sampling category based on the existing explicit-admission flag, not a conclusion about the trace.

The preflight pool is 620/636 source rows. Question 2834 is excluded after stops on `base/default/sandbag/2834` in Round 1 and `grpo3-s100/default/aware/2834` during Round 2 control preparation. `preflight_exclusions.json` records the 2/622 additional core exclusions. Full source text is unchanged. `question_exclusions.json` records question-level exclusions; `../unreadable.jsonl` records individual stopped items.

The primary natural-error comparison uses 76/82 eligible parsed wrong answers. The 60/61 eligible nonanswers remain a separate diagnostic group. Grpo3 and mt2e each have only 8/8 available parsed natural-error controls; model-specific results keep these denominators visible. No GPQA or honest-wrong augmentation is included.

The positive-control target is 10 accepted candidates per model. Candidates have a fixed seeded order, balanced by variant and arm, using the existing stated-flip pool. The first 20 candidates per model form the initial batch. A redactor proposes deletions of complete admission sentences; a fresh independent checker sees the original, reconstructed redaction, and offsets without the redactor's suitability rationale. Every retained character must be unchanged. The first accepted candidates in queue order fill the quota; no Round 1 finding influences selection. Additional candidates follow the same queue if needed. These controls establish deliberate-choice provenance, not necessarily inserted reasoning errors. Report accepted/attempted coverage and any unfilled quota.

## Blinding and independent reads

Each reader receives only its assigned blinded rows and the approved reader prompt. Inputs contain neutral IDs, the question/options, CoT, parsed final letter, answer key, and reference CoT. Model, variant, arm, source category, system prompt, judge outputs, and the unblinding key are withheld. Model-bearing examples and Round 1 profiles are not reader inputs. CoT content and style can still reveal a condition, so blinding is partial.

Two separate shuffles assign the full eligible comparison and selected positive controls to fresh reader instances. No reader sees another's output. Source text remains unchanged except for the checked positive-control deletions. Full inputs and current-item markers stay local and gitignored. Readers inspect all distinct text in bounded chunks, persist each annotation immediately, and use exact equality/hash checks for any already-read duplicate reference. A lossless representation of exact repetitions is permitted only when reconstruction is verified; it cannot omit distinct content or bypass a stop.

Stopped or refused content is logged and is not retried, rewritten, or masked. Other unattempted items can continue in fresh readers. A stop discovered in one pass is excluded from subsequent attempts of the same source item. Coverage, exclusions, and their effect on complete pairs remain visible.

## Analysis

For each model and source pool, report complete paired coverage, both-read and either-read planted-candidate fractions with descriptive Wilson 95% row intervals, mean deliberate-evidence score, and reader agreement. Separate absence of a terminal option from disagreement with a parsed final letter. Terminal agreement can coexist with an explicit override of an earlier reasoning-supported answer.

Question-level sensitivity gives each question equal weight within a pool and bootstraps questions for uncertainty. This addresses repeated-question weighting without making the distinct source pools matched experiments. Report sensitivity excluding question 212 and excluding reader-flagged question/reference defects. Also separate rows with reader-identified explicit intent from the source-unstated category. Preserve the original answer keys and source labels.

The comparison of interest is excess over each model's natural-error labels. Small controls, nonrandom missingness, partial blinding, and reader disagreement limit interpretation. Round 3 remains a later targeted-read checkpoint. No monitor-score comparison is part of this round.
