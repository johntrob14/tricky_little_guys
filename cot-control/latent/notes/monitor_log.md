# Run monitor log: mats3-migration (1x H200, $5.29/hr)

Read-only checks every ~10 min. Note: `nvidia-smi -c` is --compute-mode, not a sample count; sampling uses a shell loop instead.

## 2026-10-14 04:57 UTC
- Jobs: none running. Latest: cotsw-smoke1 killed 04:46 (10m06s); switch-smoke2 exited 0; determinism-diag killed.
- GPU: 0% util, 0% mem-bw, 0 MiB used, 78 W, no compute processes. IDLE.
- Rows: switch-smoke/generations/switch-latent: argon 14, default 10 (scratch). Nothing under /workspace/results.
- Disk: /workspace 84G/187G (45%), / 198G/500G (40%).
- Verdict: GPU idle with nothing queued (~$5.29/hr wasted while idle). Messaged main.

## 2026-10-14 04:57 UTC (addendum)
- cotsw-smoke2 (qwen3-8b default smoke, gpqa, limit 1/constraint, 2048 tok, run-dir scratch/smoke) and cotsw-tests2 started 04:57; GPU still loading (0%). Sent correction to main.

## 2026-10-14 05:08 UTC
- Jobs running: cotsw-smoke2 (11m), cotsw-tests2 (pytest, ~95%, all PASSED so far), cotsw-integ-smoke (started 05:00). New exit: think-empty-probe exited (1) at 05:04 after 3m55s.
- GPU: 1% util, 0% mem-bw, 36.2 GB used, 121 W. Two vLLM EngineCores on the GPU (19.3 GB + 16.4 GB) plus one 0.5 GB python process. Both engines were just finishing weight loading (KV cache 675k tokens, 20.6x concurrency at 32k), so no decoding yet.
- Rows: unchanged (switch-smoke argon 14, default 10). No smoke/ generations file yet.
- Disk: /workspace 45%, / 40%.
- Note: cotsw-smoke2 still loading weights 11 min after start (engine re-created, or slow startup).

