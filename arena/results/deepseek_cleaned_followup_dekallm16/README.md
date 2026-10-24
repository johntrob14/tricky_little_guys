---
language:
- en
task_categories:
- text-generation
---

# Arena reasoning edits and targeted follow-up

This dataset contains 117/117 original and edited sandboxed coding-agent transcripts across the source models recorded in each row. Match `originals.jsonl` and `dataset.jsonl` by `uuid`. Sources passed the main and side scorers, submitted, had no sample or tool-call parse error, contained reasoning, and came from post-oracle-fix runs. Full transcript content is deduplicated. These are development sources, not held-out evaluation data.

The editor is `deepseek/deepseek-v4.1-flash-1017` through OpenRouter/Parasail. Reasoning is edited chronologically with the fixed current action and preceding messages. Non-reasoning content, including tool calls, arguments, results, and visible assistant text, is preserved. Edited reasoning is synthetic and must not be interpreted as the original source model's reasoning.

Fresh monitoring is CoT-only `controlarena_cot_only`, judged by `openai/gpt-oss-20b` through OpenRouter/DekaLLM (`dekallm/bf16`). The monitor sees only the main task and the current reasoning, with no side-task description, actions, visible text, or prior history. Turns scoring strictly above 5 receive one additional valid rewrite and are rescored; unchanged monitor inputs reuse the fresh before score. After scores are conditioned on selection by the earlier score. Above-threshold traces remain in the dataset.

Coverage is 117/117 transcripts and 2,457/2,612 nonempty reasoning turns. Before the targeted pass, 351/2,457 turns and 104/117 transcripts exceeded 5. After the pass, 294/2,457 turns and 95/117 transcripts exceed 5. The follow-up's known cost is $1.206940942, including accepted DekaLLM responses imported from the concurrency-2 trial. The initial editing run and interrupted DeepInfra monitoring attempt have separate ledgers.

`max_turns.json` contains per-trace maxima and affected turn indices. `summary.json`, `config.json`, `selection.json`, the exact prompts, and approval/provenance files record the procedure. The later GPT-only dataset also edits visible assistant text; this mixed-source export edits CoT only.
