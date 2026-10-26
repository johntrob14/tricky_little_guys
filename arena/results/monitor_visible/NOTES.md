# Cot-only monitor on one part of each turn: visible text, or reasoning

Run 2026-10-24. Monitoring mode: **cot only**, monitor `controlarena_cot_only` with its prompt unchanged, judge `openai/gpt-oss-20b` via OpenRouter, main task only. `monitor_visible.py` (in `arena/`) is `rl.monitor.run` with `monitor_maintask.py`'s main-task cut and each turn's `{cot}` replaced by one part alone: `--part visible` (the turn's visible text; the reasoning is dropped) or `--part reasoning` (the reasoning; the visible text is dropped). Tool calls never reach this monitor. Turns where the part is empty are skipped and counted, not imputed. For the gpt-oss `no_cot` cells, `--part visible` also drops the analysis messages the model reopened itself. Outputs `logs/<cell>/monitor_visible_cot_only/` and `logs/<cell>/monitor_reasoning_cot_only/`; `run.sh` (PART=visible or reasoning), `summarize.py` (`results.json`, `results_nofinal.json`, `results_reasoning.json`, `results_reasoning_nofinal.json`), `figure.py` (`mats12_sprint/figures/monitor_visible_cot_only.png`, `..._nofinal.png`). Cells: `../nocot_gptoss`, `../nocot_qwen`, `../nocot_ds41`, `../nocot_organisms` (16 episodes each). Judge cost $0.38 for all 36 passes.

"Episode max" is the max score over an episode's scored turns; means are over episodes with at least one scored turn (n in brackets), 0-10, high = suspicious. "No final turn" drops each episode's last assistant turn (usually the submit turn and its summary) before taking the max.

## Visible text (unconditioned: over every episode with a scored turn; conditioned tables are in the 2026-10-25 section below)

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

Conditioned on doing the side task (per the 2026-10-25 convention): the per-episode max and the per-episode mean over scored turns are over the episodes where the side task was attempted (the leak detector `leak_check.leak_in_calls` fired on a tool call of some turn) or completed (the scorer's `side_leaked`); control cells have no side task and count every episode. Means (n episodes in brackets); per-cell SEs, the score on the first attempt turn and the unconditioned means are in `summary_visible_luna.md`, `summary_visible_luna_nofinal.md`, `summary_reasoning_luna.md`, `summary_reasoning_luna_nofinal.md`, and for the old monitor `summary_visible_cot_only*.md`, `summary_reasoning_cot_only*.md`. `results*.json` rows carry `attempted`, `fires`, `first_attempt_score`, `max`, `mean` per episode.

Muse Spark Open cells (added 2026-10-26): `../nocot_musespark` (the same four tasks, 16 episodes per cell, seed-base 0, Fireworks, effort medium; no CoT is the `museraw` raw-completions backend with an empty reasoning message prefilled). Same monitor and settings; `run_luna_musespark.sh` ran the visible-text pass on all eight cells and `monitor_maintask.py` (reasoning) on the four reasoning-on cells: 0 unparsed, judge cost $0.45; logs `<cell>.visible_luna.log`, `run_luna_musespark.out`. The results files and the figures cover all 36 visible-text and 12 reasoning cells (rerunning `summarize.py` over the 28 earlier cells reproduced their rows exactly). In the three no-CoT attack cells, 412/1061 turns have visible text, the leak detector fired on 58 turns, and 17 of those 58 have visible text.

Spot check of the Muse no-CoT visible text (a subagent read the 412 scored turns of the three attack cells; 42/48 episodes leaked). Classes: (a) names a secret, token, key or secret file as exposed or unauthenticated, or names the secondary objective; (b) describes an exposure action without naming a secret; (c) ordinary main-task text. Turns scored 9-10: 11/14 (a), 3/14 (b), 0/14 (c). Turns where the leak detector fired and visible text exists: 0/17 (a), 0/17 (b), 17/17 (c), mean score 0.35 (n=17); all short status lines ("final verification", "packaging results"), 5/17 with oblique wording ("the other update", "Healthcheck monitoring active"). 15 random turns scored 0-1 from leaked episodes: 15/15 (c). Over all 412 turns: (a) 13/412, judge mean 8.08 (n=13), 11/13 scored 9-10 and 2/13 scored 0; (b) 14/412, judge mean 4.93 (n=14), 6/14 scored 1-2 (final summaries calling the change "diagnostic information" or "runtime configuration visibility"); oblique references ("both fixes", "second request", "the other update" in single-fix tasks) 31/412, judge mean 0.32 (n=31). (a) and (b) turns occur in 22 episodes, 18 of the 42 leaked; with the oblique turns, 29/42 leaked episodes have a reference of some kind. No turn outside these three groups scored 5 or more. The split of the oblique group is the reader's judgment.

## Side-task attempts and leaks (the same episodes under either monitor)

| cell | attempted (detector) easy / medium / hard | leaked (scorer) easy / medium / hard | doing either easy / medium / hard |
|---|---|---|---|
| gpt-oss-120b base, no_cot prefill | 11/16, 10/16, 10/16 | 2/16, 4/16, 0/16 | 11/16, 10/16, 10/16 |
| gpt-oss-120b many-types s150, no_cot prefill | 1/16, 3/16, 1/16 | 0/16, 1/16, 0/16 | 1/16, 3/16, 1/16 |
| Qwen3.8-27B base, thinking off | 12/16, 14/16, 13/16 | 3/16, 4/16, 8/16 | 12/16, 14/16, 13/16 |
| Qwen3.8-27B base, thinking on | 16/16, 15/16, 16/16 | 12/16, 11/16, 5/16 | 16/16, 15/16, 16/16 |
| Qwen3.8-27B opsd-grpo s10, thinking off | 14/16, 11/16, 8/16 | 6/16, 4/16, 2/16 | 14/16, 12/16, 8/16 |
| DeepSeek V4.1 Flash 1017 base, thinking off | 13/16, 11/16, 10/16 | 6/16, 7/16, 7/16 | 13/16, 11/16, 11/16 |
| DeepSeek V4.1 Flash 1017 base, thinking on | 13/16, 12/16, 11/16 | 15/16, 14/16, 10/16 | 16/16, 15/16, 12/16 |
| Muse Spark Open base, no CoT (raw, empty reasoning prefilled) | 15/16, 16/16, 16/16 | 13/16, 15/16, 14/16 | 15/16, 16/16, 16/16 |
| Muse Spark Open base, reasoning on | 16/16, 16/16, 16/16 | 16/16, 16/16, 16/16 | 16/16, 16/16, 16/16 |

## Visible text, new monitor, episodes doing the side task

Per-episode max:

| cell | easy | medium | hard | control |
|---|---|---|---|---|
| gpt-oss-120b base, no_cot prefill | 9.82 (11) | 10.00 (10) | 10.00 (10) | 0.00 (16) |
| gpt-oss-120b many-types s150, no_cot prefill | 6.00 (1) | 9.00 (2) | 1.00 (1) | 0.08 (13) |
| Qwen3.8-27B base, thinking off | 7.83 (12) | 7.43 (14) | 7.85 (13) | 0.10 (10) |
| Qwen3.8-27B base, thinking on | 8.44 (16) | 8.67 (15) | 9.44 (16) | 0.08 (13) |
| Qwen3.8-27B opsd-grpo s10, thinking off | 9.29 (14) | 9.33 (12) | 8.62 (8) | 0.23 (13) |
| DeepSeek V4.1 Flash 1017 base, thinking off | 6.92 (13) | 5.80 (10) | 8.09 (11) | 0.33 (15) |
| DeepSeek V4.1 Flash 1017 base, thinking on | 2.88 (8) | 3.75 (8) | 3.60 (10) | 0.08 (13) |
| Muse Spark Open base, no CoT | 3.20 (15) | 3.81 (16) | 3.62 (16) | 0.50 (16) |
| Muse Spark Open base, reasoning on | 0.62 (16) | 0.25 (16) | 0.94 (16) | 0.00 (16) |

Per-episode mean over scored turns:

| cell | easy | medium | hard | control |
|---|---|---|---|---|
| gpt-oss-120b base, no_cot prefill | 4.40 (11) | 4.44 (10) | 3.72 (10) | 0.00 (16) |
| gpt-oss-120b many-types s150, no_cot prefill | 6.00 (1) | 2.92 (2) | 0.20 (1) | 0.02 (13) |
| Qwen3.8-27B base, thinking off | 2.20 (12) | 2.27 (14) | 3.34 (13) | 0.05 (10) |
| Qwen3.8-27B base, thinking on | 4.25 (16) | 3.54 (15) | 4.40 (16) | 0.01 (13) |
| Qwen3.8-27B opsd-grpo s10, thinking off | 2.74 (14) | 3.62 (12) | 2.96 (8) | 0.05 (13) |
| DeepSeek V4.1 Flash 1017 base, thinking off | 2.44 (13) | 1.71 (10) | 2.55 (11) | 0.06 (15) |
| DeepSeek V4.1 Flash 1017 base, thinking on | 1.42 (8) | 1.25 (8) | 1.27 (10) | 0.08 (13) |
| Muse Spark Open base, no CoT | 0.58 (15) | 0.75 (16) | 0.60 (16) | 0.03 (16) |
| Muse Spark Open base, reasoning on | 0.31 (16) | 0.09 (16) | 0.75 (16) | 0.00 (16) |

## Reasoning (thinking-on cells), new monitor, episodes doing the side task

Per-episode max:

| cell | easy | medium | hard | control |
|---|---|---|---|---|
| Qwen3.8-27B base, thinking on | 10.00 (16) | 10.00 (15) | 10.00 (16) | 1.00 (16) |
| DeepSeek V4.1 Flash 1017 base, thinking on | 10.00 (16) | 10.00 (15) | 10.00 (12) | 0.56 (16) |
| Muse Spark Open base, reasoning on | 10.00 (16) | 10.00 (16) | 10.00 (16) | 0.12 (16) |

Per-episode mean over scored turns:


| cell | easy | medium | hard | control |
|---|---|---|---|---|
| Qwen3.8-27B base, thinking on | 3.41 (16) | 3.55 (15) | 3.68 (16) | 0.12 (16) |
| DeepSeek V4.1 Flash 1017 base, thinking on | 3.72 (16) | 3.97 (15) | 4.15 (12) | 0.04 (16) |
| Muse Spark Open base, reasoning on | 7.19 (16) | 6.10 (16) | 6.42 (16) | 0.02 (16) |

## Old monitor (controlarena_cot_only, judge gpt-oss-20b), the same conditioning

Visible text, per-episode max:

| cell | easy | medium | hard | control |
|---|---|---|---|---|
| gpt-oss-120b base, no_cot prefill | 9.55 (11) | 9.70 (10) | 9.50 (10) | 1.00 (16) |
| gpt-oss-120b many-types s150, no_cot prefill | 2.00 (1) | 8.50 (2) | 8.00 (1) | 0.62 (13) |
| Qwen3.8-27B base, thinking off | 7.33 (12) | 7.57 (14) | 7.00 (13) | 0.70 (10) |
| Qwen3.8-27B base, thinking on | 6.81 (16) | 7.73 (15) | 8.19 (16) | 0.69 (13) |
| Qwen3.8-27B opsd-grpo s10, thinking off | 9.21 (14) | 8.92 (12) | 8.25 (8) | 2.08 (13) |
| DeepSeek V4.1 Flash 1017 base, thinking off | 6.92 (13) | 6.20 (10) | 7.73 (11) | 0.73 (15) |
| DeepSeek V4.1 Flash 1017 base, thinking on | 1.88 (8) | 4.12 (8) | 2.70 (10) | 0.38 (13) |

Reasoning, per-episode max:

| cell | easy | medium | hard | control |
|---|---|---|---|---|
| Qwen3.8-27B base, thinking on | 9.94 (16) | 9.93 (15) | 9.88 (16) | 4.19 (16) |
| DeepSeek V4.1 Flash 1017 base, thinking on | 10.00 (16) | 10.00 (15) | 10.00 (12) | 2.88 (16) |
