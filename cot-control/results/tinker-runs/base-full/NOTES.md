# Full CoT-Control (default protocol) on base gpt-oss-120b, 2026-10-23

Same command and settings as ../len2-step60-full without --model-path: greedy, top_p 1, max 25,000 tokens, "analysis channel"
wording, effort medium; gpt-oss-120b judges (1 judge error of ~1,580 calls). 12,050 rows. 12.83M output tokens (~$10.78) + 5.0M
prompt (~$1.65); judges not tallied. Raw generations and judgments gitignored, on zeus at this path.
Macro 3.6% (no meta 3.0%); suppression 7.8 (5.9), addition 2.4 (2.4), stylistic 0.6 (0.6). Paper base: 5.9% overall; Table 2 14.0 / 4.8 / 2.7.
No-constraint accuracy 51.1%. Comparison with len2 step 60, including at matched CoT length: ../../full_compare (rl/full_compare.py).
