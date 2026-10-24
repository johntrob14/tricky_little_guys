---
language:
- en
task_categories:
- text-generation
---

# Base GPT-OSS-120B Arena transcripts

This dataset pairs original sandboxed coding-agent transcripts with edited reasoning and visible assistant text. The source actor is base `openai/gpt-oss-120b`, with Tinker and OpenRouter/Parasail provenance retained per row. The source pool contains the existing 160/160 episodes and a fresh 48/48-episode batch. The existing selection contributes 158/160 eligible sources; the final combined count is in `summary.json`. Match `originals.jsonl` and `dataset.jsonl` by `uuid`.

Selection requires post-oracle-fix provenance, base-model identity, no sample-level error, nonempty reasoning, and full-content uniqueness across both batches. Main-task success, side-task success, submit, tool-call parse errors, and replay equality are not selection gates. There is no attempt-detector gate. Outcomes and errors are retained as labels; inclusion does not imply a successful or attempted side task. Deduplication ignores message/tool-call IDs and serving metadata; semantic near-duplicates are retained. These are development sources, not an independent held-out evaluation set.

The editor is `deepseek/deepseek-v4.1-flash-1017` through OpenRouter/Parasail, with medium reasoning effort and temperature 0. Each nonempty reasoning block is edited chronologically with preceding messages and the fixed current action, without future results or current-turn sibling reasoning. Message and content-block boundaries are retained. Existing cleaned exports are reused; only newly selected sources require model calls.

Monitoring is CoT-only `controlarena_cot_only`, judged by `openai/gpt-oss-20b` through OpenRouter/DekaLLM. The monitor sees the main task and current reasoning only, with no side-task description, actions, visible assistant text, or prior history. After initial rewriting, fresh scores select turns strictly above 5 for one additional valid rewrite and rescore. Remaining above-threshold traces are retained. After scores are conditioned on selection by the earlier score. `max_turns.json` contains the per-trace results.

Original-transcript baseline monitoring is incomplete because the monitor sometimes returns no score for brief reasoning. Its failed responses remain in the initial-stage cache. The followup stage independently scores the completed rewritten transcripts; its results determine targeted rewriting and the maxima in `max_turns.json`.

Nonempty turns with no valid monitor score after three attempts are retained as missing. Per-turn records keep `score: null` and `monitor_error`; exports list their indices in `unscored_monitor_turns` and use `monitor_status: complete_with_missing_scores`. Missing scores do not trigger threshold-based rewriting. `max_turns.json` reports the observed maximum among valid scores and marks `score_coverage_complete: false` when any nonempty turn is unscored; an entirely unscored trace has a null maximum. Missing scores and empty reasoning are counted separately. `summary.json` includes score coverage before and after the targeted pass. Editor failures and transport errors still stop the stage.

Visible assistant text, including intermediate text and final answers, receives one separately approved editing pass after CoT cleanup, with plausible cover stories permitted. There is no visible-text monitoring cutoff or targeted second pass. CoT and its recorded monitor inputs are preserved by that pass. Tool calls, arguments, results, system/user messages, and non-text assistant blocks are preserved throughout. Malformed calls are retained and labeled, not silently repaired. Edited text is synthetic and must not be interpreted as the source actor's original reasoning or statements.

The new batch uses the same eight approved TikTok tasks, with 6/6 easy-tier episodes per task, medium reasoning, no prefill, and seed base `2026102402`. Configuration, exact prompts, source hashes, validation records, and queue commands are recorded under `arena/results/gptoss_collect_20261024_topup` and `arena/results/gptoss_topup_cleaned_v1` in the project repository.
