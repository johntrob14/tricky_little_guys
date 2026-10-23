# Full CoT-Control (default protocol) on len2 step 60, 2026-10-23

Model: gpt-oss-120b + len2 step 60 LoRA (`tinker://b7a21354-dcbc-5b80-9bd5-f10f690ee7f1:train:0/sampler_weights/cotlen-len2-step60`),
trained only on CoT length targets (no style constraints). Protocol `default`: every eligible GPQA (445), HLE (469) and MMLU-Pro
(300) item x 9 constraints (10,836 rows) + a no-constraint baseline arm (1,214). No question overlap with the training pool (0 exact
matches by normalized text). Settings as the official `run_cceval.py`: temperature 0, top_p 1, max 25,000 tokens, "analysis
channel" wording; effort medium (template default). Judges: gpt-oss-120b, ours (official uses gpt-5-mini). Raw generations and
judgments (185M) are gitignored, on zeus at this path.

    .venv/bin/python tinker_cotcontrol.py generate --run-dir ../results/tinker-runs/len2-step60-full --protocol default \
      --condition gpt-oss-120b --model openai/gpt-oss-120b --extra-stop-ids 200012 --reasoning-term "analysis channel" \
      --model-path <len2 step 60>   # then grade / judge / analyze

Macro over 27 (dataset, constraint) cells: 18.9% (meta allowed), 12.6% (no meta). Paper, base gpt-oss-120b: 5.9% (metric variant not stated).
By group, meta allowed (no meta): suppression 40.9 (22.3), addition 13.4 (13.1), stylistic 2.35 (2.33). Paper Table 2 base: 14.0, 4.8, 2.7.
Baseline-arm accuracy 47.7% (1,214), constrained 43.9%. 23 truncated, 24 degenerate. 4.78M output tokens (~$4.01) + 5.0M prompt (~$1.65); judges not tallied.
