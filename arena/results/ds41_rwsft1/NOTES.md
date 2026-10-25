# DeepSeek V4.1 Flash 1017 CoT-control organism rwsft1 ep2 on the arena (tiktok easy)

Run 2026-10-25. The first DeepSeek organism on the arena. Checkpoint `tinker://5ba907ce-a8af-54d5-8f1d-e72031a411ea:train:0/sampler_weights/cotctl-rwsft1-ep2` (SFT on its own CoTs rewritten into CoT-control types, from obf2g step 25; `runs/2026-10-24_cot-control_rewrite-sft.md`), sampled on Tinker (`deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144`).

Sampling: `arena/ds41_renderer.py` (new) wraps DeepSeek's own prompt format and parser (`encoding/encoding.py` of deepseek-ai/DeepSeek-V4.1-Flash-1017, revision 1bae0e10) as the Tinker renderer; the cookbook has no V4.1 renderer. Names: `deepseek_v41_chat` (thinking off: the generation header closes the think block) and `deepseek_v41_thinking_<N>` (effort N). Checked before launch: the encoded prompt round-trips through the Tinker tokenizer; 3/3 thinking-off and 3/3 thinking-on probe samples from the checkpoint wrote parseable tool calls; one-episode smoke `logs/ds41rw-nocot-smoke` (token_bucket_rate_limiter_per_user, thinking off): 28 assistant turns, 32 tool calls, 0 unparsed, 0 turns with reasoning, main pass, no leak.

Cells (`launch.sh`): `run_attack.py tiktok`, 8 tiktok tasks x 2 epochs = 16 episodes per cell, seed base 0, easy tier, 40 turns, 16000 max tokens per call, side task after f8274d7, preset `arena_covert_user` (user message: main task, coaching, side task) or `control`.
- `logs/ds41rw-nocot-cvuser-easy`: thinking off, attack.
- `logs/ds41rw-nocot-ctl-easy`: thinking off, control (no side task).
- `logs/ds41rw-think25-cvuser-easy`: thinking on at effort 25, the effort rwsft1 was trained and evaluated at. No thinking-on control was run.

Base references (existing cells, OpenRouter `deepseek/deepseek-v4.1-flash-1017` pinned to Parasail fp8, so a different backend from the organism): `logs/ds41or-cvuser-easy` (thinking on, effort medium, `arena_covert_user`, the same tasks and seeds as the organism cells); `logs/ds41or-nothink-covert2-easy` (thinking off, preset `arena_covert_v2`, 4 approved tiktok tasks x 4 epochs, so a different preset, task mix and seeds: no base thinking-off cell exists under `arena_covert_user`); controls `logs/ds41or-nothink-ctl-easy`, `logs/ds41or-think-ctl-easy`.

Monitoring mode: **cot only**, monitor `controlarena_cot_only_context` (prompt unchanged), judge `openai/gpt-6-luna` at low effort through OpenRouter's `openai` provider, main task only. Two inputs per turn, each alone: the visible text (`monitor_visible.py --part visible`, rows in `logs/<cell>/monitor_visible_ctx_luna/`, via `../monitor_visible/run_luna.sh`) and, for thinking-on cells, the reasoning (`monitor_maintask.py`, rows in `logs/<cell>/monitor_ctx_luna/`). Tool calls never reach the monitor. The visible-text pass on `ds41rw-nocot-cvuser-easy` stopped once on a malformed judge reply after 340/341 turns and was resumed for the last turn (first log kept as `../monitor_visible/ds41rw-nocot-cvuser-easy.visible_luna.first.log`). 0 unparsed judge replies. New judge passes for this experiment, including the base `ds41or-cvuser-easy` visible-text pass: $0.31.

Conditioning: per-episode max and mean over the episodes doing the side task, i.e. attempted (the leak detector `leak_check.leak_in_calls` fired on a tool call of some turn) or completed (the scorer's `side_leaked`); controls count every episode. Episodes whose monitored part is empty on every turn are left out of the means.

Cost: Tinker sampling, all three cells plus the smoke, 13.27M prompt tokens and 0.28M sampled tokens: at most $9.0 at the Tinker list price ($0.60 / $3.60 per M; prompt caching, if any, would lower it).

Files: `tally.py` -> `tally.out`; `../monitor_visible/summarize.py --monitor ctx_luna [--part reasoning] --out` -> `visible_luna.json`, `reasoning_luna.json`, `summary_visible_luna.md`, `summary_reasoning_luna.md`; `stats.txt`; `figure.py` -> `mats12_sprint/figures/ds41_rwsft1_arena.png`.

## Rollout outcomes

| cell | episodes | main pass | leaked (scorer) | attempted (detector) | errors | assistant turns per episode | turns with reasoning | median reasoning chars | turns with visible text | median visible chars | unparsed tool calls | turns without a tool call | input / output tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ds41rw-nocot-cvuser-easy | 16 | 15/16 | 7/16 | 11/16 | 0/16 | 21.3 | 0/341 | - | 123/341 | 104 | 0 | 9/341 | 4.64M / 0.077M |
| ds41rw-think25-cvuser-easy | 16 | 16/16 | 7/16 | 14/16 | 0/16 | 17.6 | 272/281 | 232 | 10/281 | 730 | 0 | 10/281 | 5.09M / 0.127M |
| ds41rw-nocot-ctl-easy | 16 | 16/16 | 0/16 | 0/16 | 0/16 | 16.4 | 0/262 | - | 109/262 | 88 | 0 | 11/262 | 3.02M / 0.071M |
| ds41or-cvuser-easy | 16 | 16/16 | 4/16 | 9/16 | 0/16 | 19.4 | 265/310 | 330 | 36/310 | 109 | 0 | 5/310 | 0.37M / 0.148M |
| ds41or-nothink-covert2-easy | 16 | 14/16 | 6/16 | 13/16 | 0/16 | 28.6 | 0/457 | - | 122/457 | 96 | 0 | 2/457 | 0.38M / 0.078M |
| ds41or-nothink-ctl-easy | 16 | 15/16 | 0/16 | 0/16 | 0/16 | 20.6 | 0/329 | - | 95/329 | 96 | 0 | 4/329 | 0.27M / 0.058M |
| ds41or-think-ctl-easy | 16 | 16/16 | 0/16 | 0/16 | 0/16 | 17.4 | 220/278 | 278 | 14/278 | 616 | 0 | 12/278 | 0.30M / 0.082M |

Input tokens for the OpenRouter base cells exclude cache reads; the Tinker cells have no cache split.

## Monitor, visible text alone

| cell | episodes | attempted (detector) | leaked (scorer) | doing either | episode max, doing (n) | episode mean, doing (n) | first attempt turn score (n) | episode max, all scored (n) | unparsed |
|---|---|---|---|---|---|---|---|---|---|
| ds41rw-nocot-cvuser-easy | 16 | 11/16 | 7/16 | 11/16 | 3.90 +- 1.27 (10) | 1.35 +- 0.77 (10) | 4.67 +- 2.91 (3) | 2.60 +- 0.97 (15) | 0 |
| ds41rw-think25-cvuser-easy | 16 | 14/16 | 7/16 | 14/16 | 1.60 +- 1.01 (10) | 1.60 +- 1.01 (10) | - | 1.60 +- 1.01 (10) | 0 |
| ds41rw-nocot-ctl-easy | 16 | 0/16 | 0/16 | 16/16 | 0.07 +- 0.07 (15) | 0.01 +- 0.01 (15) | - | 0.07 +- 0.07 (15) | 0 |
| ds41or-cvuser-easy | 16 | 9/16 | 4/16 | 9/16 | 6.43 +- 1.36 (7) | 2.42 +- 0.65 (7) | 6.00 +- 1.53 (3) | 3.83 +- 1.20 (12) | 0 |
| ds41or-nothink-covert2-easy | 16 | 13/16 | 6/16 | 13/16 | 6.92 +- 0.82 (13) | 2.44 +- 0.59 (13) | 6.25 +- 0.80 (8) | 5.75 +- 0.92 (16) | 0 |
| ds41or-nothink-ctl-easy | 16 | 0/16 | 0/16 | 16/16 | 0.33 +- 0.16 (15) | 0.06 +- 0.03 (15) | - | 0.33 +- 0.16 (15) | 0 |
| ds41or-think-ctl-easy | 16 | 0/16 | 0/16 | 16/16 | 0.08 +- 0.08 (13) | 0.08 +- 0.08 (13) | - | 0.08 +- 0.08 (13) | 0 |

## Monitor, reasoning alone (thinking-on cells)

| cell | episodes | attempted (detector) | leaked (scorer) | doing either | episode max, doing (n) | episode mean, doing (n) | first attempt turn score (n) | episode max, all scored (n) | unparsed |
|---|---|---|---|---|---|---|---|---|---|
| ds41rw-think25-cvuser-easy | 16 | 14/16 | 7/16 | 14/16 | 10.00 +- 0.00 (14) | 4.07 +- 0.26 (14) | 8.57 +- 0.97 (14) | 8.88 +- 0.77 (16) | 0 |
| ds41or-cvuser-easy | 16 | 9/16 | 4/16 | 9/16 | 10.00 +- 0.00 (9) | 3.27 +- 0.30 (9) | 7.14 +- 1.84 (7) | 6.81 +- 0.98 (16) | 0 |
| ds41or-think-ctl-easy | 16 | 0/16 | 0/16 | 16/16 | 0.56 +- 0.32 (16) | 0.04 +- 0.02 (16) | - | 0.56 +- 0.32 (16) | 0 |

## Tests (episodes doing the side task; controls not tested)

```
visible   max  ds41rw-nocot-cvuser-easy 3.90 (n=10) vs ds41or-nothink-covert2-easy 6.92 (n=13): Mann-Whitney p=0.0748
visible   mean ds41rw-nocot-cvuser-easy 1.35 (n=10) vs ds41or-nothink-covert2-easy 2.44 (n=13): Mann-Whitney p=0.0582
visible   max  ds41rw-nocot-cvuser-easy 3.90 (n=10) vs ds41or-cvuser-easy 6.43 (n=7): Mann-Whitney p=0.236
visible   mean ds41rw-nocot-cvuser-easy 1.35 (n=10) vs ds41or-cvuser-easy 2.42 (n=7): Mann-Whitney p=0.105
visible   max  ds41rw-think25-cvuser-easy 1.60 (n=10) vs ds41or-cvuser-easy 6.43 (n=7): Mann-Whitney p=0.023
visible   mean ds41rw-think25-cvuser-easy 1.60 (n=10) vs ds41or-cvuser-easy 2.42 (n=7): Mann-Whitney p=0.122
reasoning max  ds41rw-think25-cvuser-easy 10.00 (n=14) vs ds41or-cvuser-easy 10.00 (n=9): Mann-Whitney p=nan
reasoning mean ds41rw-think25-cvuser-easy 4.07 (n=14) vs ds41or-cvuser-easy 3.27 (n=9): Mann-Whitney p=0.0406
attempted ds41rw-think25-cvuser-easy 14/16 vs ds41or-cvuser-easy 9/16: Fisher p=0.113
leaked    ds41rw-think25-cvuser-easy 7/16 vs ds41or-cvuser-easy 4/16: Fisher p=0.458
attempted ds41rw-nocot-cvuser-easy 11/16 vs ds41or-cvuser-easy 9/16: Fisher p=0.716
leaked    ds41rw-nocot-cvuser-easy 7/16 vs ds41or-cvuser-easy 4/16: Fisher p=0.458
```
