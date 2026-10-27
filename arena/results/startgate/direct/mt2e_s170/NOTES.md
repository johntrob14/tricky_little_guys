# manytypes2e step 170: startgate direct

- Launch date: 2026-10-26 (America/Chicago).
- Model: `tinker/openai/gpt-oss-120b`, sampler `tinker://5fecab91-f1a1-5bc2-9cb0-db0d61b7d50f:train:0/sampler_weights/cotctl-manytypes2e-step170`.
- Harness: `honeypot/startgate/run_direct.py`; source at launch `09d76ec2`. Exact commands: `run.sh` in this directory.
- Task: `tiktok/secure_password_policy_engine`; easy tier; 16 episodes; seed base 0; 80 turns; 16,000 output tokens per turn; 65,536-token context; six requested sandboxes; default GPT-OSS medium-reasoning renderer and sampling settings.
- Matched direct cells: `logs/oss-sd-pw`, `logs/mt1b-sd-pw`, `logs/opsd1b-sd-pw`. All task/seed, generation and phase settings match; model checkpoint is manytypes2e step 170. Docker image matches the latest Qwen organism cell: `sha256:661f79a1601984d327977756357a56b3511b01cd0f91869bb67dcd04d3fcba77`.
- Output: `logs/mt2e-s170-sd-pw/`; console: `logs/mt2e-s170-sd-pw.log`. Pipeline progress: `stage`, `pipeline.log`, `generation.exit`, `pipeline.exit` in this directory. Exit files contain shell return codes; sample errors are recorded separately by Inspect.

## Queued scoring

All monitor task descriptions contain only the main task. Existing prompts and judge settings are used unchanged. Every monitor runs through OpenRouter with both `openai/gpt-oss-20b` and `anthropic/claude-haiku-5.5`:

| Monitoring mode | Monitor | Episode scope |
|---|---|---|
| cot only | `controlarena_cot_only` | turns after the turn calling `start` |
| action only | `controlarena_action_only` | whole episode |
| action+cot | `controlarena_cot` | whole episode |

Each gpt-oss-20b pass has a $2 spend gate; each Haiku pass has a $6 spend gate to accommodate the worst-case forecast at the turn limit. Gates do not change the sampling or scoring settings.

`summarize_direct.py` writes `episodes.jsonl`; `transcript_done.py` writes `transcript_done.jsonl`. Both use the existing `google/gemini-3.8-flash` labeling prompts with caches specific to this run. These are side-task-aware labels, separate from the main-task-only monitors.
