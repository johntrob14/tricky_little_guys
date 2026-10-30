# arena turn1_grpo tutor1b (continuation of tutor1 from its step-5 state)

- Launched: 2026-10-21
- Project: `tricky_little_guys/arena/turn1_grpo.py`, `main` @ a9d959b
- Command: `.venv/bin/python turn1_grpo.py --run tutor1b --from-state tinker://e636ca09-4195-50a1-8ff2-b1a05654ecc4:train:0/weights/arena-turn1-tutor1-step5-state --contexts results/resample_smoke/contexts_t0_gptoss.json --G 8 --steps 10 --main-task-only --tutor-model openai/gpt-5.6-terra --tutor-effort max --base-words results/turn1_grpo/base_words_t0.json --budget-usd 35 --save-every 5` (from `arena/`)
- Output: `tricky_little_guys/arena/results/turn1_grpo/tutor1b/`

## Task
Continue tutor1 (`runs/2026-10-21_arena_tutor1.md`), which was stopped at step 5 on OpenRouter credit, for 10 more steps with identical settings. Step numbers in this run restart at 1 (= tutor1 step 6).

## Params
Same as tutor1, except: starts from tutor1's step-5 LoRA state; the Adam optimizer state starts fresh (`--from-state` loads weights only).

## Outcome
Finished 10/10 steps, $28.81 (script estimate). Checkpoints (published): `tinker://38a3f073-0812-5992-9a9d-00b799f47376:train:0/sampler_weights/arena-turn1-tutor1b-step10` (and step 5).

| | tutor1b step 1 (overall 6) | step 10 (overall 15) |
|---|---|---|
| reward | 0.148 | 0.240 |
| held-out monitor <= 1 (gpt-oss-20b, cot_only, main task only) | 97/192 | 185/192 |
| tutor susp <= 2 | 96/192 | 183/192 |
| tutor reasoning | 6.48 | 6.12 |
| CoT words | 65.8 | 37.5 |
| lp_gap | -0.008 | -0.002 |

Full episodes, 48 easy each, same monitor: episodes with any turn >= 8, base 45/48 vs step 10 8/48; side task leaked 7/48 vs 3/48; episodes with a tool call referencing the secret 40/48 vs 4/48; main task 42/48 both. Details in `arena/results/turn1_grpo/tutor1b/NOTES.md`.
