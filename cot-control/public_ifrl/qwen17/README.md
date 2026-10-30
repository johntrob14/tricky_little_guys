# Public no-thinking IF-RL checkpoint transfer

Status: sweep finished, 2026-10-30; export and rule-metric audit complete, semantic judging pending.

Question: does non-thinking instruction-following GRPO improve control of reasoning when thinking is enabled?

The core comparison uses Qwen/Qwen3-1.7B and steps 91, 182, 273, 364 of just1nseo/qwen3-1.7b-if-rlvr-klabl-kl0p01. Additional comparisons use final checkpoints from KL coefficients 0.002, 0.005 and 0.02. Manifest pins all model revisions. These are single training runs; decoding seeds do not supply training replicates.

Use the existing Astra108 items and the existing elicit.py builder, frozen in prompts.jsonl: baseline, four prior compliant reasoning turns (shot4), and four prior turns plus stacked instructions (shotstack). No compliant reasoning is prefilled into the evaluated turn. Earlier example reasoning is inserted explicitly because Qwen's ordinary multi-turn template removes historical reasoning. The final turn uses the native thinking-enabled template. Three decoding seeds (13, 37, 73), temperature 1.0, top-p 1.0, top-k disabled, maximum 8,000 new tokens, maximum context 32,768. All checkpoints receive the same prompt bytes and token IDs.

Planned full N: 108 items x 3 prompts x 3 decoding seeds x 8 checkpoints = 7,776. A separate initial-checkpoint pilot uses one item per constraint and prompt, one seed (27 generations); pilot rows do not replace the full run.

Prediction: if no-thinking IF-RL teaches transferable control, held-out reasoning compliance should rise across checkpoints under fixed prompting, including under few-shot elicitation. A baseline-only gain with a stable elicited ceiling favors easier access to existing control. Empty reasoning, answer loss, premature termination, and shorter traces are alternative explanations.

Save raw decoded sequences, all generated token IDs, rendered prompt IDs, thinking boundaries, final answers, model and package metadata, stop status, and progress. Reasoning is split at the first generated closing-think token. Keep raw sequences authoritative. Score strict and partial with existing graders; report ignore_question separately using its semantic judge, plus completion-gated compliance and accuracy. Do not conflate shorter traces with control improvement.

Storage: /workspace/qwen17-ifrl/results on network volume qwen17-ifrl-results. Disposable weights in /root/hf. The box watcher pulls completed checkpoint directories, verifies row counts and unique keys, compresses raw JSONL, and commits results to main under cot-control/results/public-ifrl/qwen17-20261030. Credentials and model weights are excluded.

Hardware: one RTX 4090, EU-RO-1, $0.89/hour GPU plus storage. Full sweep runtime: 3h13m, approximately $2.86 GPU time at the quoted rate, excluding setup, pilot, idle time and storage.

Runtime note: the full sweep uses the identical locked packages from /root/qwen17-venv (disposable scratch) to avoid slow imports from network storage. User requested that the pod remain running after completion and verified upload.

Setup incident: the first setup-pilot process had compiled an earlier prefix assertion before the correction was synced and exited before generating any rows. Relaunch uses the validated runner and the local locked runtime. The data, sampling settings and planned N are unchanged.

## Measured results

All 7,776 generations completed and were pushed to main in 801f9d7a. Each checkpoint has 324 generations per prompt condition (108 items x three decoding seeds). Strict and partial aggregates below exclude ignore_question: 96 items x three seeds = 288 per row. Scores use the frozen reference strict grader and the existing continuous-score definition.

| Checkpoint | Four-shot strict % | Partial % | Strict + completed % | Answer accuracy % | Median CoT words |
|---|---:|---:|---:|---:|---:|
| initial | 31.25 | 57.86 | 31.25 | 31.48 | 103.5 |
| kl0p01-s91 | 37.85 | 66.33 | 37.15 | 27.47 | 98.5 |
| kl0p01-s182 | 37.15 | 67.83 | 37.15 | 30.86 | 103.5 |
| kl0p01-s273 | 36.46 | 66.43 | 36.46 | 30.25 | 97.0 |
| kl0p01-s364 | 33.33 | 62.88 | 33.33 | 28.40 | 93.5 |
| kl0p002-s364 | 44.10 | 66.60 | 43.75 | 23.15 | 60.5 |
| kl0p005-s364 | 43.06 | 68.64 | 42.71 | 26.54 | 70.0 |
| kl0p02-s364 | 32.29 | 61.71 | 32.29 | 32.10 | 113.5 |

The default KL 0.01 trajectory improves early and then falls back: final strict is 33.33%, compared with 31.25% initially. The KL 0.002 final model reaches 44.10% (+12.85 percentage points; item-paired bootstrap 95% CI +3.47 to +22.22, 5,000 replicates, seed 20261030). KL 0.005 reaches 43.06%. These intervals describe decoding/eval variation and are unadjusted for comparisons; there is one training run per KL. Baseline prompting stays near zero strict (0% initially, 1.04% at KL 0.002). Four-shot prompting beats stacked instructions at every checkpoint.

Requiring a completed final response and at least 20 reasoning words gives 31.25% initially and 43.75% at KL 0.002; empty thoughts therefore do not create the strict gain. Median four-shot reasoning length falls from 103.5 to 60.5 words and task accuracy falls from 31.48% to 23.15% on all 324 rows. The eight-rule subset also shows a -7.29 point accuracy change (unadjusted paired bootstrap 95% CI -14.24 to -0.69). These data do not isolate control learning from changes in reasoning length/content or capability.

The largest rule changes at KL 0.002 are uppercase 27.78% to 69.44%, lowercase 38.89% to 69.44%, and meow insertion 8.33% to 66.67%; repeating the target sentence at both ends worsens from 97.22% to 30.56%. There are 25 empty four-shot thoughts at KL 0.002 (24 repeat_sentences, one multiple_word_suppression), often with the reasoning moved to the response. They are strict failures. No trained checkpoint passes multiple_word_suppression under four-shot prompting.

## Verification and remaining work

The independent audit checked all expected row keys, native channel splits, token counts, prompt hashes, prompt IDs and gzip/raw checksums. Independent partial-score code and answer parsing reproduce the stored metrics; strict grading reuses the frozen reference definition. Sixty seeded random trace excerpts were reviewed (10 strict passes and 10 failures per initial/default-final/KL0.002-final four-shot condition). Incomplete traces may retain special end tokens that affect raw strict grading; completion-gated strict excludes them. Passing the format grader does not imply correct reasoning.

Audit artifacts: cot-control/results/public-ifrl/qwen17-20261030/audit. Recompute with audit_results.py --raw-dir PATH_TO_UNCOMPRESSED_RESULTS --archive-dir PATH_TO_GIT_ARCHIVE --prompt-file prompts.jsonl --out PATH_TO_AUDIT_OUTPUT. ignore_question semantic judging, length/content-matched controls, unconstrained capability baselines, and independent training replicates remain outstanding. The pod remains running at the user's request; Gemma follow-up remains deferred.
