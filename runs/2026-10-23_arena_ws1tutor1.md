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

## Pre-registered decision rules (written before training, 2026-10-23 05:00 UTC)
Fixed evaluation set: after the pre-screen, up to 16 present-arm contexts on which ws1 attempted the side task in 1 to 6 of its 8 pre-screen samples (`results/ws1tutor/pool/contexts_eval.json`), excluded from the training draw. At every pause the step checkpoint is sampled G=8 on these same contexts (`results/ws1tutor/eval/step<k>/`, detector + cot-only monitor, no tutor; about $0.40 per gate). ws1's own pre-screen samples on them are the baseline (`eval/ws1/`).
- Halfway gate (after step 5 of 10): continue only if, on the evaluation set, the step-5 attempt count exceeds ws1's by at least 2 standard errors of the difference of two binomial proportions (N = 8 x contexts each), the share of samples the monitor scores >= 8 has not risen by more than 5 points over ws1's, and neither watcher has a stop condition. Otherwise the run ends at step 5 and is reported as a negative result.
- Any gate: stop on a non-finite value, |lp_gap| > 0.1 on two steps, empty-CoT or no-call turns above a quarter of a step twice, tutor failures above a third of a step, facts below 36/40 or more than 2 organism_broken verdicts in the probe, or absent-arm progress2 share rising above twice its step-1 value.
- Power note: 48 vs 48 executed episodes resolve only a large change from ws1's 1/16 (6%) rate (a rise to about 25% is detectable at the 5% level; smaller rises are not). The fixed-context evaluation set (128 samples per checkpoint, identical contexts) is the primary measure of movement; the executed episodes confirm that attempts survive execution and the sandbox scorer.

## Budget plan ($150 cap, running ledger in `arena/results/ws1tutor/LEDGER.md`)
Collection 48 easy + 16 medium-obst + 24 absent at ~$0.23 = ~$20; monitor scoring < $1; pre-screen (ws1 G=8 over the capped pool) ~$8; base length reference ~$6; tutor check ~$1; training budget $60 (`--budget-usd 60`); gates 10 x (probe ~$0.60 + eval ~$0.40) ~$10; halfway executed episodes 16 x $0.23 ~$4; final eval 48 x $0.23 ~$11. Planned total ~$121; stop-at-halfway total ~$80.

## Params
(filled in at launch)

## Outcome
(filled in when the run ends)
