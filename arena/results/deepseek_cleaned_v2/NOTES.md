# Transcript cleanup with DeepSeek

## Status

Running, resumed 2026-10-24 at 12:58 UTC in tmux session `rewrite-cleaned-v2`; `resume1.json` records the command and `resume1.log` its output. The original launch was at 10:20 UTC. Accepted replies are reused; unfinished calls use the same prompts and settings. Ethan approved the displayed editor prompt and launch with “Go ahead with full 200”; `approval.json` records its exact hash. The requested target is up to 200 unique source transcripts, bounded by eligible sources available locally.

The first pass ended at 11:29 UTC with complete rewrites exported for 85/117 sources and complete monitoring for 64/117 sources. Monitoring is CoT only, reasoning text exclusively, using `controlarena_cot_only` and judge `openai/gpt-oss-20b` through OpenRouter/DeepInfra, with the main task only. Rewriting stopped for 32/117 sources: missing response fields in 20/32, malformed JSON in 8/32, and output truncation in 4/32. Monitoring stopped for 21/117 sources: no parsed score in 12/21 and provider rate limits in 9/21. Known returned-generation cost is $3.039223314; accepted responses cover 3,951/3,995 returned generations. `pass1.json` preserves this completed-pass summary. Complete rewritten transcripts are in `dataset.jsonl`; incomplete prefixes and all responses remain in `working/`.

The completed standard post-fix attack logs contain 389/389 inventoried episodes. Content deduplication and eligibility filters yield 117/389 sources, exhausting the eligible inventory: 41/117 DeepSeek, 27/117 base Qwen through OpenRouter, 44/117 Qwen checkpoints through Tinker, and 5/117 base gpt-oss-120b through Tinker. All 117/117 pass both scorers, call submit, have no sample/tool-parse error, and contain reasoning. Structured tool-result errors are allowed and occur in 59/117 sources. The selected sources contain nonempty reasoning in 2,457/2,612 assistant turns. The source prefill is `none` for 112/117 and `no_cot` for 5/117; the latter still contain reasoning to edit. Pre-fix oracle-accessible episodes and control/honeypot/startgate conditions are excluded. Source selection does not use monitor scores. No training/evaluation split is assigned.

`inventory.json` records all 389/389 inventoried sources and the exact preparation command. `selection.json` freezes 117/276 candidates from the audited groups and additional run directories containing eligible sources; its selected content hashes equal all 117/389 eligible content hashes in the full inventory. All 117/117 frozen sources pass the preservation, reasoning-block, task-prefix, and uniqueness preflight checks. `launch.json` records the command, timestamp, code hashes, and checkout revision.

## Workflow

- `arena/rewrite_prepare.py` verifies source hashes, removes duplicates after normalizing message/tool-call IDs, and freezes original messages with a selection manifest. Additional source logs must record a revision descended from the scorer-oracle fix.
- `arena/rewrite.py` uses `deepseek/deepseek-v4.1-flash-1017` through OpenRouter, pinned to Parasail: medium thinking, temperature 0, output limit 16,000 tokens, concurrency 4. The editor receives preceding edited history, the current fixed action/visible text, and a separate `reasoning_to_edit` field. It returns complete replacement reasoning. The prompt is `arena/prompts/rewrite_reasoning.txt`; launch requires approval of its exact hash.
- Every accepted turn saves a validated prefix and locally computed character-span changes. Actions, visible text, prompts, tool results, and reasoning replay serialization are checked. Invalid replies remain recorded; they stop the affected episode. An explicit rerun reuses accepted replies and attempts failed turns again with the same settings.
- Monitoring is **CoT only, reasoning text exclusively**, monitor **`controlarena_cot_only`**, judge **`openai/gpt-oss-20b` through OpenRouter, pinned to DeepInfra**. It sees the main-task description only, without prior history, actions, visible assistant text, or tool results. Original and exact exported reasoning are scored; identical inputs reuse their score. Empty inputs are skipped, not assigned zero. Scores are diagnostic and do not select rewrites.
- `dataset.jsonl` contains one complete edited transcript per unique source, with source provenance. A monitor failure does not remove a completed edited transcript; its monitor status is recorded. Partial episodes remain in `working/<uuid>/prefix.json` and are not exported as complete episodes.

`config.json` and the copied prompt files fix inference settings and prompt hashes. `progress.json` records per-episode phases and known returned-generation costs. Raw requests, responses, accepted-reply references, edits, and monitor measurements are under `working/`. `summary.json` records export and monitoring coverage. Synthetic benchmark values may be present in raw data; these files stay local during preparation.

## Validation

Offline checks pass 10/10 tests in `arena/tests/test_rewrite.py`: action preservation, replay-text consistency, current-turn targeting, malformed/truncated/empty replies, ID-normalized deduplication, interrupted-run recovery, cached replay, exact-export monitoring with main-task-only input, and the prompt-approval gate. Tests use a fake model client and make no model calls.

The collector uses `arena/.venv/bin/python`. The inference runner uses `/home/ek/wgmn/mechtools/.venv/bin/python` from `/home/ek`, where the shared helper loads the existing API configuration. Launch command:

```sh
/home/ek/wgmn/mechtools/.venv/bin/python -u -B /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena/rewrite.py --out /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena/results/deepseek_cleaned_v2 --send
```

Output is in `run.log`; `progress.json` is the current status and cost record. The same command resumes accepted turns and retries failed requests without changing the editor settings or prompt.
