# GPT-OSS-120B source collection, 2026-10-24

Status: awaiting explicit provider approval; 0/64 episodes started. Automatic approval review requires approval to send the private task prompts and sandbox tool outputs to Parasail through OpenRouter. The launch command is prepared and has not run.

Base `openai/gpt-oss-120b` through OpenRouter, pinned to Parasail with provider fallback disabled. The endpoint reports FP4 weights. Medium reasoning effort; existing `arena_covert_v2` prompt; no reasoning prefill. This is source generation for the approximately 200 cleaned, unique transcript target. The batch contains 64 raw episodes, and eligibility is measured after completion.

All 8/8 approved TikTok tasks receive eight easy-tier episodes, with seed base `2026102401`, a 40-turn limit, and at most 16,000 generated tokens per turn. Two concurrent sandbox stacks; this process uses `ARENA_MAX_SANDBOXES=8`. The unchanged harness seeds each episode separately and creates the side-task oracle only at scoring time.

The previous normal-reasoning base collection on Tinker yielded 3/16 eligible easy-tier episodes and 0/32 eligible medium/hard episodes. This batch uses easy tier and retains the full task mix. Provider provenance is recorded separately from the Tinker sources.

Generation estimate: $1.4712 for 64 episodes, extrapolating the previous 16/16 easy-tier episodes' 3,250,069 input tokens and 57,048 output tokens at Parasail's published $0.10/M input and $0.75/M output prices, without cache discounts. Token usage can differ across providers and seeds. Prices: https://openrouter.ai/api/v1/models/openai/gpt-oss-120b/endpoints (checked 2026-10-24).

Eligibility requires both main-task and side-task success, a successful submit, no sample error or tool-call parse error, and nonempty reasoning. Full transcript content is deduplicated. Reasoning-block structure is checked for compatibility with the existing cleanup workflow. This launch performs source generation and task scoring; later cleanup starts after reporting the eligible source count.

Command: `zsh results/gptoss_collect_20261024/launch.sh` from `arena/`. Eval logs: `arena/logs/gptoss-collect-20261024-easy-s2026102401/`. Terminal log: `arena/logs/gptoss-collect-20261024-easy-s2026102401.log`. Configuration, source-file hashes, and provider-check metadata are in this directory.

Provider check: 1/1 benign requests (`Reply exactly OK.`) returned the requested answer and one nonempty reasoning block through the same Inspect/OpenRouter provider configuration. Task prompts and sandbox tool outputs were not sent in this check.
