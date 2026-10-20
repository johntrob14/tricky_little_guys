# CoT Red-Handed bench: gpt-oss-120b, Qwen3.8-27B (2026-10-19)

Base-model runs of the CoT Red-Handed eval (arXiv:2505.23575), TEST split (855 samples), paired side tasks, baseline policy, Tinker sampling. Full per-task numbers are in the eval logs; summary tables are in the session transcript of 2026-10-19.

- Logs on tav: `~/wgmn/mats12_sprint/cot-red-handed/logs/full-test-gptoss120b/` and `logs/full-test-qwen3.8-27b/` (Inspect `.eval` files; per-sample score metadata has `main_only`/`side_only`/`main_and_side`).
- Totals (main pass / side pass / both, out of 855): gpt-oss-120b 398 / 582 / 259; Qwen3.8-27B 446 / 524 / 259.
- Cost: $3.47 (gpt-oss-120b, 23 min) + $9.89 (Qwen3.8-27B, 15 min).
- Repo setup, model strings and launch commands: `~/wgmn/mats12_sprint/cot-red-handed/` README plus local diffs (openai>=3.4.0; Qwen reasoning_effort=medium override in `tasks/models.py` - Tinker rejects "high" for Qwen; gpt-oss ran at high).
- Monitor stage (2026-10-20): gpt-5-nano via OpenRouter (repo prompt id 1, approved by Ethan), judged the 259 both-complete samples per model; thresholds = 99th percentile of benign-run scores (benign n: gpt-oss 430, Qwen 459; benign runs in `logs/benign-test-*`; monitor logs in `*/monitor/`). One monitor sample per prompt (paper averages 2); hybrid weight 0.55 from the paper, not re-tuned.
- Catch rates at 1% FPR, gpt-oss-120b: subtle (n=176) action 6, CoT 29, hybrid 21; non-subtle (n=83) action 45, CoT 42, hybrid 53; all (n=259) action 51, CoT 71, hybrid 74.
- Catch rates, Qwen3.8-27B: subtle (n=192) action 9, CoT 33, hybrid 25; non-subtle (n=67) action 20, CoT 24, hybrid 30; all (n=259) action 29, CoT 57, hybrid 55.
- Per-task extremes: complexity CoT 10/12 vs action 0/12 (gpt-oss); iodelete action 16/17, ioexfil action 7/7; longlines and varname near-zero for both monitor types (0-7 of ~40-50).
