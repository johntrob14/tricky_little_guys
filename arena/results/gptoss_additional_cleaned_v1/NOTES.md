# Additional base GPT-OSS-120B sources

Selection: 5/80 eligible unique episodes from `oss-base-covert2-easy` and `gptoss-collect-20261024-easy-s2026102401`, comprising 3/16 and 2/64 respectively. The 5/5 sources are distinct from the existing 5/117 GPT-source transcripts in `deepseek_cleaned_v2`. There is nonempty reasoning on 63/70 assistant turns, in 67/67 reasoning blocks. Selection rules and frozen original messages are recorded in `selection.json` and `sources/`.

The editor is `deepseek/deepseek-v4.1-flash-1017` through OpenRouter/Parasail. Each nonempty reasoning block is edited independently with the same preceding-message history, retaining block boundaries; current-turn sibling reasoning is omitted from the editor input. The monitor scores the combined reasoning for the whole assistant turn. Monitoring is CoT-only `controlarena_cot_only`, judge `openai/gpt-oss-20b` through OpenRouter/DekaLLM, with main-task-only context and no actions, visible text, or prior history. `config.json` records the exact prompt hashes and inference settings.

The stage runs inside `../gptoss_cleaned_v1/pipeline.py`, followed by the fresh monitoring and targeted follow-up in `../gptoss_additional_followup_v1`, then the approved visible-text pass on the combined 10/10 eligible GPT sources. Offline rewrite and pipeline validation passed 37/37 tests.
