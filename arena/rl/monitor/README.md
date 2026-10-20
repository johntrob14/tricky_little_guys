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
- `monitor.jsonl`, rows `arena.monitor.v1`: run, log, sample, epoch, task, turn, monitor, scale, direction, prompt_sha256, judge_model, raw (the judge's final answer only; the score is parsed
  from this), judge_thinking (the thinking, kept apart and never parsed), score (null if unparsed/skipped), reasoning, unparsed, skipped, empty_cot, attempts, tokens {in,out}, cost_usd, sidecar.
- `summary.json`: per episode max and mean over parsed turns, counts of unparsed/empty-cot/skipped turns, per-task means, total cost, and `judge` (backend, model, max_tokens, price with its verified status).

A reply is unparsed unless it holds one clean `<score>` in range (agreeing duplicates are fine); one retry, then recorded unparsed. No regex fallback; unparsed turns are left out of means.
A reply cut off inside the thinking (no `</think>`) has an empty answer and is therefore unparsed.
Resume: rows already in `monitor.jsonl` (same run, log file, sample, epoch, turn, monitor, judge model) are skipped, so unparsed rows are not retried again. Samples are keyed by log file, so the
same (sample, epoch) in two `.eval` files of one run dir are both kept. A different judge model does not resume from another judge's rows, and the summary covers only the current judge's rows.
If the judge raises mid-run, no further calls are made, rows already written stay, `summary.json` is still written, the result carries `error`, and the exit code is 1.

## Judge
- Default: DeepSeek V4.1 Flash on Tinker, model id exactly `deepseek-ai/DeepSeek-V4.1-Flash:peft:262144` (Tinker refuses the bare name). A thinking model: the model's own chat template (prompt ends inside `<think>`),
  temperature 0, stop on EOS, `--max-tokens` 6000 (thinking included), reply split at `</think>`. Needs `TINKER_API_KEY` in the environment.
- `--judge-model openai/gpt-oss-20b` (or `-120b`) stays selectable on Tinker (cookbook renderer, low reasoning effort).
- `--judge-backend openrouter`: OpenAI-compatible `POST https://openrouter.ai/api/v1/chat/completions` through httpx (no new dependency), temperature 0, 429/5xx retried 3 times. The key comes from
  `OPENROUTER_API_KEY` or `OPENROUTER_API_KEY=...` in `~/.config/sauce/openrouter.env`; it is never printed or logged, and a missing key stops the run before any call. The default model is
  `deepseek/deepseek-v4.1-flash` (the entry "DeepSeek: DeepSeek V4.1 Flash" in OpenRouter's model list, matched by name; that it serves the same weights as the Tinker checkpoint is **unverified**).
  Override with `--judge-model`; any model not in `pricing.py` is costed at an UNVERIFIED upper bound.

## Prices (`pricing.py`), USD per million tokens
| backend / model | prefill | cached prefill | sample | source |
|---|---|---|---|---|
| tinker DeepSeek-V4.1-Flash | 0.60 | 0.12 | 3.60 | thinkingmachines.ai/tinker/pricing, fetched 2026-10-20 |
| tinker gpt-oss-20b / 120b | 0.18 / 0.33 | 0.036 / 0.066 | 0.45 / 0.84 | same page |
| openrouter deepseek/deepseek-v4.1-flash | 0.02832 | 0.000566 | 0.12 | openrouter.ai/api/v1/models, fetched 2026-10-20 |
| unlisted tinker / openrouter model | 1.695 / 3.0 | no credit | 4.215 / 15.0 | UNVERIFIED conservative upper bound; flagged in `--dry-run` and `summary.json` |

Cached prefill is never credited (upper bound). The Tinker page was read as text; its columns were taken to be prefill, cached prefill, sample, train (the order that matches the GPT-OSS rows already used in
`self-oversight/common.py`). The OpenRouter list price is the listed per-token price; the provider OpenRouter routes to may bill differently.

## Running
`python -m rl.monitor.run LOGS --dry-run` prints backend, model, price (with verified status), prompt count, token estimate and worst-case cost, spends nothing. A live run needs `--budget-usd`
(finite, > 0) and `--allow-live` for either backend, refuses if the worst-case forecast (every prompt at `--max-tokens` output) exceeds the cap, reserves worst-case cost before each call and stops
(rows kept) when the cap would be passed. Token counts for the forecast are characters/3, not a tokenizer. Flags: `--judge-backend {tinker,openrouter}`, `--judge-model`, `--max-tokens`, `--monitor`,
`--out`, `--sidecars`, `--concurrency`, `--strip-prefill` (the monitor sees each turn's reasoning without the prefill the run was sampled with; logs from `run_attack.py --prefill`; default output `<logs>/monitor_cont`).

`reward_hook.py` is an inert reward component (raises unless `enabled=True`), not wired into training.

## Status
Nothing live was run, on either backend. `judges.tinker_judge` is written against the Tinker sampling API as used in `cot-control-deepseek/tinker_cotcontrol.py` but has never been called (a thinking-model
reply format, `</think>` split, is untested against a real reply); `judges.openrouter_judge` is tested only with a fake transport. Tests use fake judges. `rl.py` did not exist when this was written, so there is no subcommand: use `python -m rl.monitor.run` (`main()` in `run.py`).
