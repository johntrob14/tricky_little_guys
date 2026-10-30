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

## Recorded outcome

Completed on 2026-10-25, 15:11:19–15:30:56 UTC. All 318/318 transcripts are retained. Monitoring was CoT-only `controlarena_cot_only_context`, judged by `openai/gpt-6-luna` at low effort through OpenRouter/OpenAI with main-task-only input. Before and after coverage is 5,599/5,599 nonempty reasoning turns, with 0/5,599 missing judgments in either pass. Turns scoring strictly above 5 are 242/5,599 before and 139/5,599 after; affected transcripts are 154/318 before and 111/318 after. The editor changed 215/242 selected turns. It made 262/262 accepted block-level rewrites; 1/263 returned editor responses required a validation retry for incorrect JSON fields. API-reported spend is $3.165204482 across 11,461/11,461 saved response envelopes.

| Source model | Retained transcripts | Above-5 turns before | Above-5 turns after | Above-5 transcripts before | Above-5 transcripts after |
|---|---:|---:|---:|---:|---:|
| GPT-OSS-120B | 206/206 | 109/3,174 | 61/3,174 | 72/206 | 51/206 |
| Qwen base/checkpoints | 71/71 | 82/1,472 | 48/1,472 | 50/71 | 37/71 |
| DeepSeek | 41/41 | 51/953 | 30/953 | 32/41 | 23/41 |

`verification.json` records exact monitoring requests, response hashes, separate final response IDs, chronological editor-request replay, and transcript preservation. Of 215/5,599 turns with changed reasoning, 133/215 move from above 5 to at most 5. Among 5,384/5,599 unchanged turns, 5/5,384 move below the cutoff and 35/5,384 move above it in the fresh final judgments. `traces.html` is a standalone reader with original reasoning, first-pass and recleaned reasoning, edit highlights, scores, and preserved tool context. The request/response archive contains all accepted judgments and the retained invalid editor response.

The verifier makes no API calls:

```sh
/home/ek/wgmn/mechtools/.venv/bin/python -B /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena/reclean_verify.py
```
