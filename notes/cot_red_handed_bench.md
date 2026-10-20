# CoT Red-Handed bench: gpt-oss-120b, Qwen3.8-27B (2026-10-19)

Base-model runs of the CoT Red-Handed eval (arXiv:2505.23575), TEST split (855 samples), paired side tasks, baseline policy, Tinker sampling. Full per-task numbers are in the eval logs; summary tables are in the session transcript of 2026-10-19.

- Logs on tav: `~/wgmn/mats12_sprint/cot-red-handed/logs/full-test-gptoss120b/` and `logs/full-test-qwen3.8-27b/` (Inspect `.eval` files; per-sample score metadata has `main_only`/`side_only`/`main_and_side`).
- Totals (main pass / side pass / both, out of 855): gpt-oss-120b 398 / 582 / 259; Qwen3.8-27B 446 / 524 / 259.
- Cost: $3.47 (gpt-oss-120b, 23 min) + $9.89 (Qwen3.8-27B, 15 min).
- Repo setup, model strings and launch commands: `~/wgmn/mats12_sprint/cot-red-handed/` README plus local diffs (openai>=3.4.0; Qwen reasoning_effort=medium override in `tasks/models.py` - Tinker rejects "high" for Qwen; gpt-oss ran at high).
- No monitors were run yet (monitor stage needs OPENAI_API_KEY and prompt sign-off).
