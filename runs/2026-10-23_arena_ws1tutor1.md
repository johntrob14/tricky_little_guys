# arena turn1_grpo ws1tutor1 (tutor-reward GRPO on frozen contexts, from John's Qwen ws1 organism)

- Status: preparing (2026-10-23). Collection of the organism's episodes running; bug review of the pipeline running; training not launched.
- Project: `tricky_little_guys/arena/turn1_grpo.py`, `main` @ 71e26c45 + uncommitted (pause gate `--pause-every`, resume `--start-step`/`--spent-start`; `decisive_ctx.py --miss-mid`; `pool_finalize.py --org-samples/--min-org-attempts`; new `organism_probe.py`), committed with the run.
- Output: `arena/results/turn1_grpo/ws1tutor1/`; collection and pool under `arena/results/ws1tutor/`, logs `arena/logs/ws1-ctl-{easy,medium-obst,absent}`.

## Task
Ethan's tutor-reward GRPO (`runs/2026-10-22_arena_ws8tutor1.md`, `runs/2026-10-23_arena_qwentutor1.md`) applied to `warmstart-qwen-ws1-organism-ep1` (Qwen3.8-27B LoRA, `arena/results/warmstart/qwen-ws1-organism/checkpoints.json`), the Qwen port of the ws8 SFT recipe. Under the control preset ws1 completed the side task in 1/16 episodes (`logs/sniff-qwen-ws1-final`, scorer `side_leaked`), main task 15/16. Question: does tutor-reward GRPO raise the completion rate while the CoT stays quiet (cot-only monitor, tutor suspiciousness) and the model stays healthy?

## Plan
1. Episodes of ws1, control preset: easy x48 (`--epochs 6 --seed-base 100`), medium with `--obstacles` x32, `--absent` x32. Cost measured on the sniff: $0.23 per episode.
2. `monitor_maintask.py` cot-only scores on every turn; `decisive_ctx.py --drop-submit --skip-oracle-reads --miss-mid 0.33 0.5`.
3. Pre-screen: `resample_ctx.py` with ws1 at G=8 over the pool; `pool_finalize.py --org-samples ... --min-org-attempts 1 --max-prompt-tokens 16000` keeps only contexts where ws1 attempts in at least one of 8 samples (plus absent contexts). Base-Qwen G=8 resample over the kept contexts for the length reference.
4. Tutor check on ~40 organism samples (progress separates detector-fire samples from the rest).
5. Training: 16 contexts x G=8 x 10 steps, `--pause-every 1` (a hard pause after every step = every 10%), `--save-every 1`, weights as ws8tutor1 plus `--tutor-quiet-weight 0.3` as qwentutor1, `--len-lambda 0.04`, training budget $70.
6. At each pause: `organism_probe.py` on the step checkpoint (compared with the ws1 probe), then the gate workflow (two watchers, skeptics) recommends GO / STOP; at step 5 (halfway) also 16 executed control-preset episodes of the checkpoint. The run is given up at the halfway gate unless completion signals have moved up and nothing is degrading.
7. Final eval: 48 control-preset episodes of the final checkpoint vs the 48 easy reference episodes, same seeds.

## Params
(filled in at launch)

## Outcome
(filled in when the run ends)
