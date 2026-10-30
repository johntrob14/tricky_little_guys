# Checked positive controls

The selected controls are 40/40 existing stated flips, with 10/10 per model. `selection.json` records the source IDs, coverage, and fixed-queue selection. These controls establish an original deliberate-choice admission; they are not validated examples of inserted reasoning errors.

| Model | Attempted / initial candidate pool | Completed redactions / attempted | Accepted / independently checked | Selected / accepted |
|---|---:|---:|---:|---:|
| base | 25/343 | 25/25 | 11/17 | 10/11 |
| grpo3-s100 | 39/459 | 38/39 | 12/37 | 10/12 |
| mt1b-s150 | 20/79 | 20/20 | 18/20 | 10/18 |
| manytypes2e-s170 | 40/202 | 40/40 | 10/30 | 10/10 |

Candidate order is fixed by seed 20261028 and balanced across variant and arm. Rejected candidates are retained without repair. Selection takes the earliest accepted candidates, after every earlier eligible candidate has a settled redactor/checker decision. The unselected mt2e `02b` proposals are 5/40 completed candidates after the selected cutoff; they have no independent suitability decision and are not used.

Redactors remove only complete qualifying admission sentences and record Python character offsets. Checkers receive original text, its reconstructed redaction, and offsets without the redactor's rationale. Suspicious reasoning alone is not a rejection reason. All retained characters must be unchanged. `*_reconstruction.jsonl` records hashes and character counts; full originals and redactions are local, gitignored inputs.

The grpo3 reader stopped on `grpo3-s100/default/aware/2834` (1/39 attempted candidates). Stopped content was not retried or masked. That question is excluded after separate stops across Rounds 1 and 2. Question-level and core-pool exclusions are recorded in `../question_exclusions.json` and `../preflight_exclusions.json`.

The scientific prompts were approved before use. Native Codex subagents performed redaction and independent checking; the exact underlying inherited model is not exposed by the agent interface. No external paid judging or new target-model sampling was performed.
