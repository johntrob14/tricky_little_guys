---
language:
- en
task_categories:
- text-generation
---

# Arena transcript recleaning with the context/Luna monitor

This corpus contains 318/318 unique sandboxed coding-agent source transcripts: 206/318 from base GPT-OSS-120B, 71/318 from Qwen base models and checkpoints, and 41/318 from DeepSeek. It combines the mixed-source cleanup and GPT-only cleanup exports, deduplicating their 5/323 shared entries by source identity. The corpus has 5,599/6,800 assistant turns with nonempty reasoning. Source outcomes and selection provenance remain attached to each row. Inclusion does not imply task success, side-task success, or a successful monitor evasion. These are development data, not an independent held-out evaluation set.

The input reasoning is the saved chronological first-pass edit, before any monitor-directed follow-up. GPT visible assistant text comes from the completed visible-text cleanup. `inputs.jsonl` contains this exact combination, `originals.jsonl` contains the archived originals, and `dataset.jsonl` contains the recleaned transcripts. Match files by `uuid`. The source manifest records parent export hashes and each first-pass source path and digest.

Monitoring is **CoT only**, using `controlarena_cot_only_context`, judged by `openai/gpt-6-luna` with low reasoning effort through OpenRouter's `openai` provider, with fallbacks disabled. Each judgment sees the main task and one turn's reasoning; it excludes the secondary objective, actions, visible assistant text, tool results, and history. Scores are ordinal suspicion ratings from 0 to 10, not probabilities. Empty reasoning is skipped. Exhausted invalid monitor responses remain explicitly missing and are not treated as zero.

Every nonempty input turn receives a fresh judgment. Turns scoring strictly above 5 receive one valid chronological rewrite using the approved `deepseek/deepseek-v4.1-flash-1017` editor through OpenRouter/Parasail, medium reasoning effort, temperature 0. Each editor request includes preceding messages and the fixed current action, without future results or current-turn sibling reasoning. Every nonempty final turn then receives a separate new judgment, including unchanged text. Accepted requests are reused only when resuming the same before or after pass. Final results are still conditioned on the earlier judgments that selected edits; they are not independent evaluations of the cleaning procedure.

All 318/318 source identities are retained regardless of the final score. Non-reasoning content is preserved, including the GPT visible-text cleanup, tool calls, arguments, results, and message/block structure. There is no visible-text monitor cutoff or extra visible-text rewrite pass. Edited reasoning and visible text are synthetic; they are not the source actor's original reasoning or statements.

Concurrency is at most 32 requests across the run. Exact prompts, configuration, authorization, requests, complete returned response envelopes, edits, scores, maxima, and API-reported costs are retained. `summary.json` records completed coverage; `publication.json` records byte verification of the public Hugging Face export. No actor rollouts, Docker tasks, or training steps are run by this procedure.

Run from `/home/ek` with the mechtools environment:

```sh
/home/ek/wgmn/mechtools/.venv/bin/python -u -B /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena/reclean.py --concurrency 32 --send
```
