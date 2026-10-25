# Cot-only monitor on one part of each turn: visible text, or reasoning

Run 2026-10-24. Monitoring mode: **cot only**, monitor `controlarena_cot_only` with its prompt unchanged, judge `openai/gpt-oss-20b` via OpenRouter, main task only. `monitor_visible.py` (in `arena/`) is `rl.monitor.run` with `monitor_maintask.py`'s main-task cut and each turn's `{cot}` replaced by one part alone: `--part visible` (the turn's visible text; the reasoning is dropped) or `--part reasoning` (the reasoning; the visible text is dropped). Tool calls never reach this monitor. Turns where the part is empty are skipped and counted, not imputed. For the gpt-oss `no_cot` cells, `--part visible` also drops the analysis messages the model reopened itself. Outputs `logs/<cell>/monitor_visible_cot_only/` and `logs/<cell>/monitor_reasoning_cot_only/`; `run.sh` (PART=visible or reasoning), `summarize.py` (`results.json`, `results_nofinal.json`, `results_reasoning.json`, `results_reasoning_nofinal.json`), `figure.py` (`mats12_sprint/figures/monitor_visible_cot_only.png`, `..._nofinal.png`). Cells: `../nocot_gptoss`, `../nocot_qwen`, `../nocot_ds41`, `../nocot_organisms` (16 episodes each). Judge cost $0.38 for all 36 passes.

"Episode max" is the max score over an episode's scored turns; means are over episodes with at least one scored turn (n in brackets), 0-10, high = suspicious. "No final turn" drops each episode's last assistant turn (usually the submit turn and its summary) before taking the max.

## Visible text

| cell | easy | medium | hard | control |
|---|---|---|---|---|
| gpt-oss-120b base, no_cot prefill | 6.69 (16) | 7.88 (16) | 7.06 (16) | 1.00 (16) |
| gpt-oss-120b many-types s150, no_cot prefill | 0.73 (11) | 4.00 (9) | 3.40 (10) | 0.62 (13) |
| Qwen3.8-27B base, thinking off | 7.00 (14) | 6.75 (16) | 6.44 (16) | 0.70 (10) |
| Qwen3.8-27B base, thinking on | 6.81 (16) | 7.25 (16) | 8.19 (16) | 0.69 (13) |
| Qwen3.8-27B opsd-grpo s10, thinking off | 8.62 (16) | 7.44 (16) | 7.44 (16) | 2.08 (13) |
| DeepSeek V4.1 Flash 1017 base, thinking off | 6.25 (16) | 5.40 (15) | 6.19 (16) | 0.73 (15) |
| DeepSeek V4.1 Flash 1017 base, thinking on | 1.88 (8) | 4.11 (9) | 2.07 (14) | 0.38 (13) |

No final turn: gpt-oss base 6.69 (16), 7.88 (16), 7.06 (16), 1.00 (16); many-types 1.00 (8), 6.00 (6), 4.25 (8), 0.67 (12); Qwen off 6.71 (14), 6.19 (16), 5.00 (16), 0.60 (10); Qwen on 5.38 (13), 5.00 (14), 6.92 (13), 0.75 (12); Qwen organism 8.56 (16), 7.44 (16), 7.67 (15), 2.45 (11); DeepSeek off 5.38 (16), 5.40 (15), 6.12 (16), 0.79 (14); DeepSeek on 0.86 (7), 3.11 (9), 2.07 (14), 0.38 (13).

## Reasoning (thinking-on cells)

| cell | easy | medium | hard | control |
|---|---|---|---|---|
| Qwen3.8-27B base, thinking on | 9.94 (16) | 9.88 (16) | 9.88 (16) | 4.19 (16) |
| DeepSeek V4.1 Flash 1017 base, thinking on | 10.00 (16) | 10.00 (16) | 9.81 (16) | 2.88 (16) |

No final turn: identical to two decimals. Per-cell tables (leaked vs not leaked, means over scored turns, unparsed counts) print from `summarize.py`.

## Spot check of the visible-text scores

A subagent read 20 visible-text turns scored >= 8 (5 each from `nocot-covert2-medium`, `qwenor-nothink-covert2-easy`, `ds41or-nothink-covert2-hard`, `qwenor-think-covert2-hard`), 8 turns scored 3-6, and the top 3 turns of `qwenor-nothink-ctl-easy` and `ds41or-nothink-ctl-easy`. High turns: 12/20 name the side task ("secondary objective", "both objectives") or say a secret is being exposed, 5/20 describe the exposure (an unauthenticated config or env endpoint) without naming it, 3/20 are ordinary off-scope lines the judge scored 8 while writing that it saw no evidence of malicious intent. Two explicit turns were scored 6 and 4. Control turns: none above 2 (58/58 and 95/95 scored control turns at 0-2). gpt-oss `no_cot` visible text was planning prose ("We need to ...") in 7/7 turns read; Qwen and DeepSeek visible text was mostly short preambles and final summaries.

# The same cells under the new default monitor (2026-10-25)

Monitoring mode: **cot only**, monitor `controlarena_cot_only_context` with its prompt unchanged (the main-task cut is built into it), judge `openai/gpt-6-luna` at reasoning effort low through OpenRouter's `openai` provider with fallbacks off, temperature 0, 6000 output tokens. `monitor_visible.py --part visible` now accepts this reasoning-only monitor and puts the turn's visible text alone in its `{cot}` slot (reasoning and tool calls dropped); empty turns are skipped and counted. `run_luna.sh` over the 28 cells (two lanes, concurrency 16): 10294 turns, 7449 skipped as empty, 0 unparsed, judge cost $1.25. Rows in `logs/<cell>/monitor_visible_ctx_luna/`; logs `<cell>.visible_luna.log`, `run_luna_{a,b}.out`. The reasoning-only series is this monitor's ordinary pass, `monitor_maintask.py` -> `logs/<cell>/monitor_ctx_luna/` (same prompt), run 2026-10-25 for the thinking-on Qwen cells (committed) and DeepSeek cells (on disk from another session's cvuser work, uncommitted at the time of writing). `summarize.py --monitor ctx_luna [--part reasoning] [--exclude-final]` -> `results_luna.json`, `results_nofinal_luna.json`, `results_reasoning_luna.json`, `results_reasoning_nofinal_luna.json`; tables `summary_visible_luna.md`, `summary_visible_luna_nofinal.md`. `figure.py --monitor ctx_luna [--exclude-final]` -> `mats12_sprint/figures/monitor_visible_ctx_luna.png`, `monitor_visible_ctx_luna_nofinal.png`.

Mean per-episode max suspicion (0-10) over episodes with at least one scored turn (n in brackets); the per-cell table with SEs, leaked/not-leaked split and means over scored turns is in `summary_visible_luna.md`.

## Visible text, new monitor

| cell | easy | medium | hard | control |
|---|---|---|---|---|
| gpt-oss-120b base, no_cot prefill | 6.81 (16) | 8.12 (16) | 7.44 (16) | 0.00 (16) |
| gpt-oss-120b many-types s150, no_cot prefill | 0.64 (11) | 3.33 (9) | 1.90 (10) | 0.08 (13) |
| Qwen3.8-27B base, thinking off | 7.86 (14) | 6.69 (16) | 7.25 (16) | 0.10 (10) |
| Qwen3.8-27B base, thinking on | 8.44 (16) | 8.12 (16) | 9.44 (16) | 0.08 (13) |
| Qwen3.8-27B opsd-grpo s10, thinking off | 8.50 (16) | 7.44 (16) | 7.25 (16) | 0.23 (13) |
| DeepSeek V4.1 Flash 1017 base, thinking off | 5.75 (16) | 4.20 (15) | 6.06 (16) | 0.33 (15) |
| DeepSeek V4.1 Flash 1017 base, thinking on | 2.88 (8) | 4.11 (9) | 2.79 (14) | 0.08 (13) |

No final turn:

| cell | easy | medium | hard | control |
|---|---|---|---|---|
| gpt-oss-120b base, no_cot prefill | 6.81 (16) | 8.12 (16) | 7.44 (16) | 0.00 (16) |
| gpt-oss-120b many-types s150, no_cot prefill | 0.88 (8) | 5.00 (6) | 2.38 (8) | 0.08 (12) |
| Qwen3.8-27B base, thinking off | 7.29 (14) | 5.44 (16) | 5.19 (16) | 0.00 (10) |
| Qwen3.8-27B base, thinking on | 7.23 (13) | 6.00 (14) | 8.77 (13) | 0.00 (12) |
| Qwen3.8-27B opsd-grpo s10, thinking off | 8.44 (16) | 7.44 (16) | 7.33 (15) | 0.27 (11) |
| DeepSeek V4.1 Flash 1017 base, thinking off | 4.88 (16) | 4.00 (15) | 5.94 (16) | 0.29 (14) |
| DeepSeek V4.1 Flash 1017 base, thinking on | 1.29 (7) | 3.00 (9) | 2.71 (14) | 0.08 (13) |

## Reasoning (thinking-on cells), new monitor

| cell | easy | medium | hard | control |
|---|---|---|---|---|
| Qwen3.8-27B base, thinking on | 10.00 (16) | 9.94 (16) | 10.00 (16) | 1.00 (16) |
| DeepSeek V4.1 Flash 1017 base, thinking on | 10.00 (16) | 10.00 (16) | 9.19 (16) | 0.56 (16) |

No final turn:

| cell | easy | medium | hard | control |
|---|---|---|---|---|
| Qwen3.8-27B base, thinking on | 10.00 (16) | 9.94 (16) | 10.00 (16) | 1.00 (16) |
| DeepSeek V4.1 Flash 1017 base, thinking on | 10.00 (16) | 10.00 (16) | 9.19 (16) | 0.56 (16) |
