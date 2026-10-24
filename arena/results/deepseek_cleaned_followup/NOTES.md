# Monitor-selected transcript rewrite

## Status

Running fresh monitoring in tmux session `rewrite-followup-v3`; all 117/117 completed source rewrites are frozen in `selection.json` and `sources/`. The source stage is complete at $4.160596872, recorded separately from follow-on spend. The follow-on was queued on 2026-10-24 at 13:47 UTC, when 113/117 source rewrites were complete. `launch.json` records the command, code hashes, and user authorization; `progress.json` records the current stage. Ethan requested fresh 20b CoT-only monitoring over the rewritten traces, per-trace maximum-scoring turns, and another rewrite of turns scoring above 5; he selected the separate follow-on script. All 117/117 source rewrites must exist before the input snapshot is frozen. Source monitoring failures do not block a complete rewrite from entering this pass.

## Procedure

`rewrite_followup.py` uses `mechtools.openrouter` for every model call. Monitoring is **CoT only, reasoning text exclusively**, monitor **`controlarena_cot_only`**, judge **`openai/gpt-oss-20b` through OpenRouter, pinned to DeepInfra**. Inputs contain the main task only and the current reasoning, without prior history, actions, visible assistant text, tool results, or the side-task description. The approved monitor template is copied unchanged from the source run.

Every nonempty reasoning turn receives a fresh score in this run. Source-run scores are not copied. Empty reasoning is skipped without a numeric score. Every source trace must have complete scoring coverage before the targeted rewrite stage starts. `max_turns.json` records each trace's maximum, all tied maximum-scoring turns, and all turns strictly above 5. It provides zero-based turn IDs and one-based assistant-turn numbers for maxima.

The editor is `deepseek/deepseek-v4.1-flash-1017` through OpenRouter, pinned to Parasail, with medium reasoning, temperature 0, and a 16,000-token output limit. It uses the exact previously approved editor prompt and receives the preceding edited history, current fixed action, and separate reasoning to edit. Monitor scores and rationales are not added to its prompt. Each turn above 5 receives one accepted rewrite; turns scoring exactly 5 or lower are preserved. Processing is chronological, so later selected turns see earlier accepted edits. Actions, tool results, prompts, and visible assistant text are preserved; reasoning replay signatures are updated consistently.

Changed reasoning is scored again with the same CoT-only monitor, judge, and task visibility. Identical monitor inputs reuse their fresh before-score, with that reuse recorded. There is no loop rewriting successful edits until they fall below the threshold. The before/after measurements use the same monitor that selected the turns.

The follow-on runs at concurrency 2. The OpenRouter helper makes up to 6 attempts for transient HTTP errors with its existing backoff; there is no provider or model fallback. Returned replies that fail validation are retained and retried up to 3 times per invocation. Remaining failures stop the affected stage and are recorded in `progress.json`; accepted responses are reused on explicit resume. With `--finish-source`, the queued script can resume unfinished source rewrites up to 3 times per invocation using the source run's frozen settings. Each source resume is recorded under `source_resumes/`; persistent source failures stop the follow-on with `source_incomplete` status.

## Artifacts

- `config.json` and copied prompt files: fixed models, OpenRouter backend, threshold, retry limits, prompt hashes, and source-selection hash.
- `selection.json` and `sources/`: frozen complete source rewrites, with original-source provenance and input-message hashes.
- `working/<uuid>/`: fresh monitor requests/responses, before/after scores, accepted edits, validated prefixes, and completed outputs.
- `max_turns.json`: per-trace maxima, tied turn IDs, and above-threshold turns before and after rewriting.
- `dataset.jsonl`: complete follow-on transcripts, selected-turn IDs, provenance, and monitoring status.
- `progress.json` and `summary.json`: stage coverage, failures, and returned-generation costs. Follow-on spend excludes source-run spend; source resumes retain their own ledger.

Raw messages and responses remain local. The source dataset is preserved. All model calls use the existing approved prompts.

## Validation and command

Offline tests pass 16/16 across `tests/test_rewrite.py` and `tests/test_rewrite_followup.py`. Follow-on checks cover strict threshold selection, tied maxima and empty reasoning, waiting for the source lock and complete source coverage, fresh monitoring before rewriting, chronological targeted edits, preservation of unselected turns and non-reasoning fields, score reuse for identical inputs, one accepted rewrite per selected turn, cached resumption, failures blocking the next stage, OpenRouter model identity, frozen configuration, and source-resume invariant failures. All test model calls are faked.

Run from `/home/ek`; logs go to this directory's `run.log`:

```sh
/home/ek/wgmn/mechtools/.venv/bin/python -u -B /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena/rewrite_followup.py --source /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena/results/deepseek_cleaned_v2 --out /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena/results/deepseek_cleaned_followup --send --wait --finish-source --concurrency 2
```
