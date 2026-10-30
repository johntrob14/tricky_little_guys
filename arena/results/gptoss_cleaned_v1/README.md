---
language:
- en
task_categories:
- text-generation
---

# Base GPT-OSS-120B Arena transcripts

This dataset pairs original sandboxed coding-agent episodes with edited reasoning and visible assistant text for model-organism research. The source actor is base `openai/gpt-oss-120b`. Source provenance includes Tinker and OpenRouter/Parasail runs and is retained in each row. The prepared pool contains 10/160 eligible unique base-model episodes across eight approved TikTok tasks.

`originals.jsonl` contains original messages; `dataset.jsonl` contains edited messages. Match rows by `uuid`. Source selection requires main-task and side-task scorer success, a submit call, no sample or tool-call parse error, nonempty reasoning, post-oracle-fix provenance, and unique transcript content. Ordinary tool-result errors are allowed. Deduplication ignores message/tool-call IDs and serving metadata; it does not remove semantic near-duplicates.

The editor is `deepseek/deepseek-v4.1-flash-1017` through OpenRouter/Parasail. Reasoning blocks are edited independently in chronological order, retaining message and content-block boundaries. The editor receives preceding messages and the fixed current action, without later results. Current-turn sibling reasoning is not supplied to a block's editor call. Both initial and targeted CoT rewriting use the same approved prompt.

Monitoring is CoT-only `controlarena_cot_only`, judged by `openai/gpt-oss-20b` through OpenRouter/DekaLLM, with main-task-only context and no action, visible assistant text, or prior history. After fresh scoring, turns scoring strictly above 5 receive one additional rewrite pass and are rescored. Scores after this pass are conditioned on selection by the earlier score. A remaining score above 5 does not remove a transcript. `max_turns.json` and `summary.json` retain the results.

Visible assistant text, including final answers, receives one separate editing pass after the CoT work. Plausible cover stories are permitted by the approved editing instructions. This pass makes no monitor calls and uses no monitoring cutoff or targeted second pass. It preserves CoT byte-for-byte, so the recorded CoT monitor inputs remain unchanged. Tool calls, arguments, tool results, system/user messages, and non-text assistant blocks are preserved throughout cleanup. Text blocks may become empty, but their boundaries are retained.

The edited text is synthetic and should not be interpreted as the original actor's reasoning or statements. These development sources are not an independent held-out evaluation set. Exact prompts, configurations, validation results, and source eval logs are recorded under `arena/` in the project repository.
