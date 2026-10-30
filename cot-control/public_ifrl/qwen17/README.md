# Public no-thinking IF-RL checkpoint transfer

Status: prepared, 2026-10-30.

Question: does non-thinking instruction-following GRPO improve control of reasoning when thinking is enabled?

The core comparison uses Qwen/Qwen3-1.7B and steps 91, 182, 273, 364 of just1nseo/qwen3-1.7b-if-rlvr-klabl-kl0p01. Additional comparisons use final checkpoints from KL coefficients 0.002, 0.005 and 0.02. Manifest pins all model revisions. These are single training runs; decoding seeds do not supply training replicates.

Use the existing Astra108 items and the existing elicit.py builder, frozen in prompts.jsonl: baseline, four prior compliant reasoning turns (shot4), and four prior turns plus stacked instructions (shotstack). No compliant reasoning is prefilled into the evaluated turn. Earlier example reasoning is inserted explicitly because Qwen's ordinary multi-turn template removes historical reasoning. The final turn uses the native thinking-enabled template. Three decoding seeds (13, 37, 73), temperature 1.0, top-p 1.0, top-k disabled, maximum 8,000 new tokens, maximum context 32,768. All checkpoints receive the same prompt bytes and token IDs.

Planned full N: 108 items x 3 prompts x 3 decoding seeds x 8 checkpoints = 7,776. A separate initial-checkpoint pilot uses one item per constraint and prompt, one seed (27 generations); pilot rows do not replace the full run.

Prediction: if no-thinking IF-RL teaches transferable control, held-out reasoning compliance should rise across checkpoints under fixed prompting, including under few-shot elicitation. A baseline-only gain with a stable elicited ceiling favors easier access to existing control. Empty reasoning, answer loss, premature termination, and shorter traces are alternative explanations.

Save raw decoded sequences, all generated token IDs, rendered prompt IDs, thinking boundaries, final answers, model and package metadata, stop status, and progress. Reasoning is split at the first generated closing-think token. Keep raw sequences authoritative. Score strict and partial with existing graders; report ignore_question separately using its semantic judge, plus completion-gated compliance and accuracy. Do not conflate shorter traces with control improvement.

Storage: /workspace/qwen17-ifrl/results on network volume qwen17-ifrl-results. Disposable weights in /root/hf. The box watcher pulls completed checkpoint directories, verifies row counts and unique keys, compresses raw JSONL, and commits results to main under cot-control/results/public-ifrl/qwen17-20261030. Credentials and model weights are excluded.

Hardware: one RTX 4090, EU-RO-1, $0.89/hour GPU plus storage. No measured runtime estimate is available before the pilot.
