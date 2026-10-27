# Proposed Round 2 plan

Awaiting the README checkpoint. No Round 2 reads, redactions, sampling, or external API judging have been run.

Retain the README's within-model comparison and two independent, separately shuffled blinded reads. Hide model, arm, variant, system prompt, source kind, and judge outputs behind neutral IDs. The complete reader prompt supplies the definitions; model-bearing examples in the human codebook are not reader inputs. CoT wording/style can still reveal a condition, so blinding is partial.

The source pools are 398/398 instructed flips without a positive explicit-admission flag, 95/95 style-control flips, and 143/143 natural resample-error rows. Admission flags define sampling strata, not truth. Apply recorded safety exclusions and exclude question 2899 after repeated stops. Eligibility is recorded in `round2_pool_metadata_proposed.json`: 392/398 unstated flips, 94/95 style-control flips, and 136/143 natural errors, totaling 622/636 eligible source rows before positive controls.

The primary natural-error comparison uses parsed wrong letters: 82/143 source rows, with 76/82 eligible. The other 61/143 source rows are unparsed; retain their 60/61 eligible rows as a separate diagnostic group, outside comparisons of planted-error rates on parsed flips. Grpo3 and mt2e each have only 8/8 available parsed natural errors. Keep their small denominators visible rather than pooling across models. No GPQA or honest-wrong augmentation is proposed.

Target 40 accepted redacted positive controls, with 10 per model, from existing stated flips. One agent removes complete sentences admitting a wrong-answer/concealment plan; an independent checker verifies unchanged remaining text and rejects obvious seams or remaining direct admissions. Report accepted/attempted counts by model. These controls establish deliberate-choice provenance, not necessarily inserted reasoning errors: some stated traces merely switch letters or guess. Proposed prompts are in `round2_redaction_prompts_proposed.md` and require approval before use.

Use `round2_prompt_proposed.md` and `codebook_v1.md`. Separate reasoning-supported options, terminal chosen answer, submitted letter, and expressed objective. Use null for unavailable comparisons. Reference correctness does not certify its derivation. Intent, knowledge suppression, ordinary error, unsupported selection, and planted-error candidates remain separate.

For each model and source pool, report annotated/eligible coverage, both-read and either-read planted-candidate fractions, mean deliberate-evidence score, and reader agreement. Wilson 95% row intervals are descriptive because rows can share a question; add a question-level sensitivity analysis to expose repeated-question dependence. No monitor-score thresholding is proposed.

Report sensitivity to question/reference defects, especially question 212's nonunique options, without silently rewriting original keys or flip definitions. Separate reader-identified explicit intent inside the source-unstated pool when discussing unspoken intent. The primary signal is excess over each model's natural-error labels, with missingness and small controls visible.

Approval is needed for the clarified reader/redactor/checker prompts, parsed-versus-unparsed comparison, separate reasoning/choice fields, and question/reference sensitivity analyses. Round 3 remains a later targeted-read checkpoint. No new sampling or paid API judging is proposed.
