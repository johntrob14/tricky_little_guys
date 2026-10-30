# OPSD vs EI overnight run — plan
Start: 2026-10-15T04:00:54Z. Hard stop 15:00:54Z.
Deadlines: 12:00:54Z (8h) no new training; 14:00:54Z (10h) stop jobs, REPORT.md, copy /workspace locally; 14:45:54Z (10.75h) stop every pod.

Local code: ~/code/opsd (synced to /workspace/opsd). Local run log: ~/code/opsd/run_log.md (pushed to /workspace/run_log.md).
Pod: created by ~/code/opsd/poll_h200.py (log poll.log, pod id in POD_ID). Name opsd-h200, 1xH200 SXM US-CA-2, $5.29/hr (secure).

## Stages (planned durations after pod obtained; T = pod time)
| Stage | What | Plan |
|---|---|---|
| P0 | (local, during wait) spec extraction, data prep, trainer code, CPU prototype | before pod |
| P1 | Pod setup: venv on container disk, vLLM, download base 49B + SDF adapter + EI adapters, merge SDF | 60 min |
| P2 | GPU prototype of full OPSD loop on Qwen3-1.7B (sleep mode) ; Nemotron load check in vLLM + HF trainer | 30 min (parallel with P1 download) |
| S0 | Headroom gate: SDF-only ± teacher lines, 4 contexts x 128 BCB | 40 min |
| S1 | OPSD 3000 rollouts, rounds of 128, eval 64 held-out each 500 | 4 h (calibrate after round 1) |
| S2 | Eval: OPSD/EI1/EI4 main grid (4x128), all 4 ckpts robustness (15x64) | 2 h |
| R | REPORT.md, copy back | 20 min |
Cut rule: a stage >50% over plan → cut rollouts first, then robustness grid; report every cut.

## Pivot (user, 04:35Z): if no H200 by 04:45:45Z, rent 2x H100 SXM (US-CA-2, vol1, 500 GB disk).
2x H100 = 2x80 GB. bf16 base (~100 GB) still fits only once across both GPUs, so the sleep-mode swap design stays:
vLLM tensor_parallel_size=2; HF trainer device_map="auto" split over both GPUs (naive model parallel). Throughput roughly that of 1x H200.
