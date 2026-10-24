# DeepSeek transcript cleanup

## Status

Prepared; no model calls launched. Source-pool selection and approval of the exact editor prompt are pending. The requested target is up to 200 unique source transcripts, bounded by eligible sources available locally.

The audited post-fix DeepSeek corpus and the additional `logs/pilot-ds-openended/` run contain 52/52 inventoried episodes. Content deduplication and the pilot eligibility rules yield 24/52 candidates (516/558 assistant turns contain nonempty reasoning). Allowing structured tool-result errors while retaining both scorer passes, submit, and no sample/tool-parse error yields 41/52 candidates (953/1,029 turns contain reasoning). Pre-fix oracle-accessible episodes are excluded. Source selection does not use monitor scores. No training/evaluation split is assigned.

## Workflow

- `arena/rewrite_prepare.py` verifies source hashes, removes duplicates after normalizing message/tool-call IDs, and freezes original messages with a selection manifest. Additional source logs must record a revision descended from the scorer-oracle fix.
- `arena/rewrite.py` uses `deepseek/deepseek-v4.1-flash-1017` through OpenRouter, pinned to Parasail: medium thinking, temperature 0, output limit 16,000 tokens, concurrency 4. The editor receives preceding edited history, the current fixed action/visible text, and a separate `reasoning_to_edit` field. It returns complete replacement reasoning. The prompt is `arena/prompts/rewrite_reasoning.txt`; launch requires approval of its exact hash.
- Every accepted turn saves a validated prefix and locally computed character-span changes. Actions, visible text, prompts, tool results, and reasoning replay serialization are checked. Invalid replies remain recorded; they stop the affected episode. An explicit rerun reuses accepted replies and attempts failed turns again with the same settings.
- Monitoring is **CoT only, reasoning text exclusively**, monitor **`controlarena_cot_only`**, judge **`openai/gpt-oss-20b` through OpenRouter, pinned to DeepInfra**. It sees the main-task description only, without prior history, actions, visible assistant text, or tool results. Original and exact exported reasoning are scored; identical inputs reuse their score. Empty inputs are skipped, not assigned zero. Scores are diagnostic and do not select rewrites.
- `dataset.jsonl` contains one complete edited transcript per unique source, with source provenance. A monitor failure does not remove a completed edited transcript; its monitor status is recorded. Partial episodes remain in `working/<uuid>/prefix.json` and are not exported as complete episodes.

`config.json` and the copied prompt files fix inference settings and prompt hashes. `progress.json` records per-episode phases and known returned-generation costs. Raw requests, responses, accepted-reply references, edits, and monitor measurements are under `working/`. `summary.json` records export and monitoring coverage. Synthetic benchmark values may be present in raw data; these files stay local during preparation.

## Validation

Offline checks pass 10/10 tests in `arena/tests/test_rewrite.py`: action preservation, replay-text consistency, current-turn targeting, malformed/truncated/empty replies, ID-normalized deduplication, interrupted-run recovery, cached replay, exact-export monitoring with main-task-only input, and the prompt-approval gate. Tests use a fake model client and make no model calls.

The collector uses `arena/.venv/bin/python`. The inference runner uses `/home/ek/wgmn/mechtools/.venv/bin/python` from `/home/ek`, where the shared helper loads the existing API configuration. The exact launch command is recorded when the source pool and prompt are approved.
