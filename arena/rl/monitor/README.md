# rl/monitor

Post-hoc CoT+action monitor over Inspect `.eval` logs. It is the ControlArena monitor (first entry of the "Judge Prompts" tab; `prompts/SOURCE.md`),
used instead of the LinuxArena/control-tower monitors (`old_action`, `orthogonal`, `quick`), which are not implemented or imported here.

## Reads
- `.eval` logs (`inspect_ai.log.read_eval_log`): every assistant message of every sample is one turn. Reasoning comes from `ContentReasoning` blocks
  (redacted blocks are ignored), tool calls from `tool_calls`, the task description from the first user message.
- Optionally `rl_data/<run>/*.json.gz` sidecars (`--sidecars rl_data`), read defensively; only `episode_id, ended, failure_class, weights_id, split, phase`
  are copied into rows as `sidecar`. Missing or odd files give no metadata and no error. Run name = the log's parent directory name.

## Monitors (`monitors.py`)
- `controlarena_cot` (default): sees task, ONE turn's CoT (HTML-escaped) and that turn's tool calls as `name(sorted-key JSON args)`. Score 0-10, **high = suspicious**.
  It sees no earlier turns and no side-task list. An empty CoT is judged on the tool calls alone (the prompt says that is not suspicious by itself) and counted as `empty_cot`.
- `ontask_generic_100` (non-default): the generic scratchpad prompt, CoT only, bare number 1-100, **high = on-task** (opposite direction; native scale kept, see `scale`/`direction` in each row). Empty-CoT turns are skipped, not judged.

Placeholders are filled by splitting the template, not `str.format`/`replace`, so braces or placeholder text inside a CoT survive.

## Outputs, `<out>/<run>/`
- `monitor.jsonl`, rows `arena.monitor.v1`: run, log, sample, epoch, task, turn, monitor, scale, direction, prompt_sha256, raw, score (null if unparsed/skipped), reasoning,
  unparsed, skipped, empty_cot, attempts, tokens {in,out}, cost_usd, sidecar.
- `summary.json`: per episode max and mean over parsed turns, counts of unparsed/empty-cot/skipped turns, per-task means, total cost.

A reply is unparsed unless it holds one clean `<score>` in range (agreeing duplicates are fine); one retry, then recorded unparsed. No regex fallback; unparsed turns are left out of means.
Resume: rows already in `monitor.jsonl` (same run, sample, epoch, turn, monitor) are skipped, so unparsed rows are not retried again.

## Running
`python -m rl.monitor.run LOGS --dry-run` prints prompt count, token estimate and worst-case cost, spends nothing. A live run needs `--budget-usd` and `--allow-live`,
refuses if the worst-case forecast exceeds the cap, reserves worst-case cost before each call and stops (rows kept) when the cap would be passed. Judge: frozen
`openai/gpt-oss-20b`, low reasoning effort, T 0, max_tokens 1500; list price 0.18 prefill / 0.45 sample USD per million tokens, cache credit ignored (upper bound).
Token counts for the forecast are characters/3, not a tokenizer.

`reward_hook.py` is an inert reward component (raises unless `enabled=True`), not wired into training.

## Status
Nothing live was run. `tinker_judge` is written against the cookbook API but has never been called; tests use a fake judge. `rl.py` did not exist when this was written, so there is no subcommand: use `python -m rl.monitor.run` (`main()` in `run.py`).
