---
language:
- en
task_categories:
- text-generation
---

# Base GPT-OSS-120B Arena transcripts

This dataset pairs original sandboxed coding-agent transcripts with edited reasoning and visible assistant text for model-organism research. The source actor is base `openai/gpt-oss-120b`; provenance includes Tinker and OpenRouter/Parasail runs. Source selection yields 158/160 unique episodes from the post-oracle-fix collection. Nonempty reasoning, readable source records, no sample-level error, and full-content deduplication are required. Main-task success, side-task success, submit, tool-call parse errors, and exact replay equality are not selection gates. There is no attempt-detector gate. Original outcomes and errors remain explicit metadata; inclusion does not imply a successful or attempted side task. These are development sources, not an independent held-out evaluation set.

`originals.jsonl` contains original messages and `dataset.jsonl` contains edited messages, matched by `uuid`. Full-content deduplication ignores message/tool-call IDs and serving metadata; no semantic near-duplicate filtering is used. Cleanup of 10/158 sources is reused from the original queue, and the other 148/158 receive the same approved procedure. Previously accepted editor responses are reused by verified request/response hashes.

The editor is `deepseek/deepseek-v4.1-flash-1017` through OpenRouter/Parasail, with medium reasoning effort and temperature 0. Each nonempty reasoning block is edited chronologically with preceding messages and the fixed current action, without future results or current-turn sibling reasoning. Message and content-block boundaries are retained.

Monitoring is CoT-only `controlarena_cot_only`, judged by `openai/gpt-oss-20b` through OpenRouter/DekaLLM, using main-task-only context and no actions, visible assistant text, or prior history. After the initial rewrite, fresh scores select turns strictly above 5 for one additional valid rewrite and rescore. Remaining above-threshold traces are retained. These after scores are conditioned on selection by the earlier score. `max_turns.json` retains per-trace maxima and above-threshold turns.

Visible assistant text, including intermediate text and final answers, receives one separately approved editing pass after CoT cleanup. Plausible cover stories are permitted. There is no visible-text monitoring cutoff or targeted second pass. CoT and its recorded monitor inputs remain unchanged by this pass. Tool calls, arguments, tool results, system/user messages, and non-text assistant blocks are preserved throughout. Malformed tool calls remain labeled in the source data and are not silently repaired.

The edited text is synthetic and must not be interpreted as the source actor's original reasoning or statements. Exact prompts, configurations, provenance, and validation results are retained under `arena/results/gptoss_expanded_v1` in the project repository. The new stages use eight concurrent episodes, up to six transport attempts, and up to three attempts to obtain a schema-valid response. Invalid responses are retained and failures are logged; only the first valid response is accepted for each editing pass.
