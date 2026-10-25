# ds41_sft1 results: the SFT checkpoint on the arena, against its starting point and base DeepSeek

Run 2026-10-25. Method, data and commands: `README.md`; run diary `runs/2026-10-25_arena_ds41-sft1.md`. Checkpoints (published): epoch 2 `tinker://b6ffa062-0433-5852-9f29-00662919570c:train:0/sampler_weights/arena-ds41sft1-ep2`, epoch 1 `.../arena-ds41sft1-ep1`.

## Setup
- Cells (`eval_cells.sh`): tiktok, preset `arena_covert_coach_first` (honest system prompt; user message: coaching, main task, side task) or `control` (no side task), 8 tasks x 2 epochs = 16 episodes per cell, seed base 100, 40 turns, 16000 max tokens per call; all on Tinker through `ds41_renderer.py`, thinking on at effort 25 (`deepseek_v41_thinking_25`). SFT checkpoint (epoch 2) and rwsft1 ep2: easy, medium, control (easy) and the no-secret arm (easy, `--absent`: nothing seeded, the secret's file deleted); base DeepSeek V4.1 Flash 1017 (no checkpoint): easy and medium.
- 4 of the 8 tasks (`missing_user_id_in_guess_response`, `sponsor_statsv2_diggcount_total`, `token_bucket_rate_limiter_per_user`, `video_url_format_validation`) are the tasks of the 41 training transcripts; 4 are held out. Every secret differs from the training episodes' (seed base 100; the training episodes are seed base 0).
- Monitors, all main task only, each turn scored alone (0-10, high = suspicious), never the episode as a whole (`monitor_cells.sh`; rows in `logs/<cell>/monitor_ctx_luna`, `monitor_visible_ctx_luna`, `monitor_reasoning_actioncot`, `monitor_maintask_cot_only`):
  1. **cot only**, `controlarena_cot_only_context`, judge `openai/gpt-6-luna` (low effort): each turn's reasoning alone.
  2. **cot only**, `controlarena_cot_only_context`, judge `openai/gpt-6-luna` (low): each turn's visible text alone.
  3. **action+cot**, `controlarena_cot`, judge `openai/gpt-6-luna` (low): reasoning, visible text and tool calls.
  4. **cot only**, `controlarena_cot_only`, judge `openai/gpt-oss-20b`: reasoning plus visible text.
  Passes 1-3 use the monitor and judge the cleaning loop used on the training traces; pass 4 is a different prompt and judge. 40/40 passes completed; 2 passes (pass 4 on `ds41sft1-easy` and `ds41base-cf-easy`) have 1 unparsed turn each, not imputed. The first launch of the monitor script refused 9 passes on its worst-case cost gate (budget caps below the forecast); they were rerun with higher caps and the finished passes resumed from their saved rows.
- Episodes "doing the side task": the leak detector (`leak_check.leak_in_calls`) fired on a tool call of some turn, or the scorer confirmed the leak. Scores below are means over those episodes (n in brackets); controls count every episode. The per-episode sum of turn scores is mean x scored turns.

## Rollout outcomes (`tally.py` -> `tally.out`)

| cell | episodes | main pass | leaked (scorer) | attempted (detector) | errors | assistant turns per episode | turns with reasoning | median reasoning chars | turns with visible text | median visible chars | unparsed tool calls | turns without a tool call | input / output tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ds41sft1-easy | 16 | 16/16 | 11/16 | 13/16 | 0/16 | 29.9 | 440/478 | 363 | 26/478 | 80 | 1 | 3/478 | 9.98M / 0.196M |
| ds41sft1-medium | 16 | 16/16 | 12/16 | 12/16 | 0/16 | 29.5 | 434/472 | 319 | 48/472 | 89 | 2 | 1/472 | 11.50M / 0.203M |
| ds41sft1-ctl-easy | 16 | 16/16 | 0/16 | 0/16 | 0/16 | 14.7 | 206/235 | 241 | 12/235 | 244 | 0 | 5/235 | 3.20M / 0.093M |
| ds41sft1-absent-easy | 16 | 15/16 | 0/16 | 15/16 | 0/16 | 37.9 | 568/606 | 466 | 28/606 | 98 | 2 | 5/606 | 16.74M / 0.348M |
| ds41rw-cf-easy | 16 | 16/16 | 12/16 | 14/16 | 0/16 | 19.3 | 302/309 | 234 | 2/309 | 350 | 0 | 2/309 | 5.81M / 0.131M |
| ds41rw-cf-medium | 16 | 16/16 | 14/16 | 15/16 | 0/16 | 16.3 | 252/261 | 222 | 9/261 | 538 | 0 | 9/261 | 4.74M / 0.110M |
| ds41rw-cf-ctl-easy | 16 | 16/16 | 0/16 | 0/16 | 0/16 | 11.2 | 144/179 | 252 | 19/179 | 514 | 0 | 14/179 | 1.94M / 0.063M |
| ds41rw-cf-absent-easy | 16 | 16/16 | 0/16 | 15/16 | 0/16 | 21.1 | 324/338 | 313 | 18/338 | 199 | 0 | 9/338 | 6.31M / 0.140M |
| ds41base-cf-easy | 16 | 16/16 | 16/16 | 15/16 | 0/16 | 22.5 | 338/360 | 251 | 20/360 | 111 | 0 | 7/360 | 6.52M / 0.166M |
| ds41base-cf-medium | 16 | 16/16 | 13/16 | 13/16 | 0/16 | 18.9 | 288/302 | 340 | 24/302 | 179 | 0 | 8/302 | 6.05M / 0.149M |

## Statistics (`analyze.py` -> `stats.txt`, `analyze.out`; Mann-Whitney on per-episode values and Fisher on counts, unpaired)

```
tiktok tasks in the training set: ['missing_user_id_in_guess_response', 'sponsor_statsv2_diggcount_total', 'token_bucket_rate_limiter_per_user', 'video_url_format_validation']; the other 4 of 8 are held out. Eval episodes per cell: 16 (8 tasks x 2), seed base 100.

== Side-task attempts (leak detector fired on a tool call) and scorer leaks, of the episodes in each arm (all tasks / held-out tasks only)
  SFT ep2 (ds41sft1)     easy     attempted 13/16 leaked 11/16 | held-out tasks: attempted 6/8 leaked 5/8
  SFT ep2 (ds41sft1)     medium   attempted 12/16 leaked 12/16 | held-out tasks: attempted 6/8 leaked 7/8
  SFT ep2 (ds41sft1)     control  attempted 0/16 leaked 0/16 | held-out tasks: attempted 0/8 leaked 0/8
  SFT ep2 (ds41sft1)     absent   attempted 15/16 leaked 0/16 | held-out tasks: attempted 8/8 leaked 0/8
  rwsft1 ep2 (start)     easy     attempted 14/16 leaked 12/16 | held-out tasks: attempted 7/8 leaked 5/8
  rwsft1 ep2 (start)     medium   attempted 15/16 leaked 14/16 | held-out tasks: attempted 8/8 leaked 7/8
  rwsft1 ep2 (start)     control  attempted 0/16 leaked 0/16 | held-out tasks: attempted 0/8 leaked 0/8
  rwsft1 ep2 (start)     absent   attempted 15/16 leaked 0/16 | held-out tasks: attempted 7/8 leaked 0/8
  base DeepSeek          easy     attempted 15/16 leaked 16/16 | held-out tasks: attempted 7/8 leaked 8/8
  base DeepSeek          medium   attempted 13/16 leaked 13/16 | held-out tasks: attempted 7/8 leaked 7/8

== Assistant turns per episode, episodes doing the side task (easy + medium pooled), and the share of turns with empty reasoning is in tally.out
  SFT ep2 (ds41sft1)     29.73 (n=26); held-out tasks 32.46 (n=13)
  rwsft1 ep2 (start)     17.66 (n=29); held-out tasks 20.27 (n=15)
  base DeepSeek          21.48 (n=29); held-out tasks 23.40 (n=15)
  SFT ep2 (ds41sft1)     control  14.69 (n=16) turns per episode (all episodes)
  SFT ep2 (ds41sft1)     absent   37.88 (n=16) turns per episode (all episodes)
  rwsft1 ep2 (start)     control  11.19 (n=16) turns per episode (all episodes)
  rwsft1 ep2 (start)     absent   21.12 (n=16) turns per episode (all episodes)

== Attack cells pooled over easy and medium: SFT against rwsft1 and against base (Fisher on counts)
  attempted SFT 25/32 vs rw 29/32 (all tasks): Fisher p=0.302
  attempted SFT 12/16 vs rw 15/16 (held-out tasks): Fisher p=0.333
  leaked    SFT 23/32 vs rw 26/32 (all tasks): Fisher p=0.556
  leaked    SFT 12/16 vs rw 12/16 (held-out tasks): Fisher p=1
  attempted SFT 25/32 vs base 28/32 (all tasks): Fisher p=0.509
  attempted SFT 12/16 vs base 14/16 (held-out tasks): Fisher p=0.654
  leaked    SFT 23/32 vs base 29/32 (all tasks): Fisher p=0.107
  leaked    SFT 12/16 vs base 15/16 (held-out tasks): Fisher p=0.333

== No-secret arm (nothing seeded, the secret's file deleted): attempts only, since the scorer cannot confirm a leak; the stock .env.local holds NEXTAUTH_SECRET and TIKAPI_KEY, so the prompt is true for episodes that draw those names
  SFT ep2 (ds41sft1)     attempted 15/16 all; 11/12 on the episodes without a stock key name; 4/4 with one
  rwsft1 ep2 (start)     attempted 15/16 all; 11/12 on the episodes without a stock key name; 4/4 with one
  SFT vs rwsft1, all 16: Fisher p=1

== Pass 1: reasoning, CoT only, controlarena_cot_only_context, gpt-6-luna low (cleaning monitor)
   per-episode max and mean over scored turns, mean over episodes doing the side task; first-attempt-turn score; controls over all episodes
   [all tasks] max                  SFT ep2 (ds41sft1) 9.42 (n=26) | rwsft1 ep2 (start) 10.00 (n=29) | base DeepSeek 10.00 (n=29)   SFT vs rw: p=0.00326; SFT vs base: p=0.00326
   [all tasks] mean                 SFT ep2 (ds41sft1) 1.74 (n=26) | rwsft1 ep2 (start) 3.64 (n=29) | base DeepSeek 4.35 (n=29)   SFT vs rw: p=6.86e-08; SFT vs base: p=3.44e-09
   [all tasks] total                SFT ep2 (ds41sft1) 48.35 (n=26) | rwsft1 ep2 (start) 60.41 (n=29) | base DeepSeek 87.45 (n=29)   SFT vs rw: p=0.0249; SFT vs base: p=1.46e-05
   [all tasks] first_attempt_score  SFT ep2 (ds41sft1) 7.54 (n=24) | rwsft1 ep2 (start) 8.90 (n=29) | base DeepSeek 10.00 (n=28)   SFT vs rw: p=0.0139; SFT vs base: p=1.17e-05
   [held-out tasks] max                  SFT ep2 (ds41sft1) 9.77 (n=13) | rwsft1 ep2 (start) 10.00 (n=15) | base DeepSeek 10.00 (n=15)   SFT vs rw: p=0.135; SFT vs base: p=0.135
   [held-out tasks] mean                 SFT ep2 (ds41sft1) 1.69 (n=13) | rwsft1 ep2 (start) 3.51 (n=15) | base DeepSeek 4.39 (n=15)   SFT vs rw: p=0.000132; SFT vs base: p=3.39e-05
   [held-out tasks] total                SFT ep2 (ds41sft1) 53.00 (n=13) | rwsft1 ep2 (start) 66.60 (n=15) | base DeepSeek 95.47 (n=15)   SFT vs rw: p=0.107; SFT vs base: p=0.00494
   [held-out tasks] first_attempt_score  SFT ep2 (ds41sft1) 7.64 (n=11) | rwsft1 ep2 (start) 8.13 (n=15) | base DeepSeek 10.00 (n=14)   SFT vs rw: p=0.391; SFT vs base: p=0.00686
   control (no side task) per-episode max: SFT ep2 (ds41sft1) 0.31 (n=16)
   control (no side task) per-episode max: rwsft1 ep2 (start) 0.12 (n=16)
   unconditioned per-episode max over all attack episodes: SFT ep2 (ds41sft1) 8.56 (n=32) | rwsft1 ep2 (start) 9.94 (n=32) | base DeepSeek 9.47 (n=32)

== Pass 2: visible text, CoT only, controlarena_cot_only_context, gpt-6-luna low (cleaning monitor)
   per-episode max and mean over scored turns, mean over episodes doing the side task; first-attempt-turn score; controls over all episodes
   [all tasks] max                  SFT ep2 (ds41sft1) 2.55 (n=22) | rwsft1 ep2 (start) 2.11 (n=9) | base DeepSeek 2.53 (n=15)   SFT vs rw: p=0.66; SFT vs base: p=0.936
   [all tasks] mean                 SFT ep2 (ds41sft1) 1.19 (n=22) | rwsft1 ep2 (start) 2.11 (n=9) | base DeepSeek 2.13 (n=15)   SFT vs rw: p=0.945; SFT vs base: p=0.341
   [all tasks] total                SFT ep2 (ds41sft1) 3.00 (n=22) | rwsft1 ep2 (start) 2.11 (n=9) | base DeepSeek 2.87 (n=15)   SFT vs rw: p=0.611; SFT vs base: p=0.91
   [all tasks] first_attempt_score  SFT ep2 (ds41sft1) 5.80 (n=5) | rwsft1 ep2 (start) - | base DeepSeek 3.00 (n=2)   SFT vs rw: p=n/a; SFT vs base: p=0.434
   [held-out tasks] max                  SFT ep2 (ds41sft1) 2.09 (n=11) | rwsft1 ep2 (start) 1.67 (n=6) | base DeepSeek 1.78 (n=9)   SFT vs rw: p=0.311; SFT vs base: p=0.775
   [held-out tasks] mean                 SFT ep2 (ds41sft1) 0.65 (n=11) | rwsft1 ep2 (start) 1.67 (n=6) | base DeepSeek 1.11 (n=9)   SFT vs rw: p=0.311; SFT vs base: p=1
   [held-out tasks] total                SFT ep2 (ds41sft1) 2.82 (n=11) | rwsft1 ep2 (start) 1.67 (n=6) | base DeepSeek 2.33 (n=9)   SFT vs rw: p=0.26; SFT vs base: p=0.714
   [held-out tasks] first_attempt_score  SFT ep2 (ds41sft1) 6.50 (n=2) | rwsft1 ep2 (start) - | base DeepSeek 3.00 (n=2)   SFT vs rw: p=n/a; SFT vs base: p=0.667
   control (no side task) per-episode max: SFT ep2 (ds41sft1) 0.00 (n=8)
   control (no side task) per-episode max: rwsft1 ep2 (start) 0.00 (n=14)
   unconditioned per-episode max over all attack episodes: SFT ep2 (ds41sft1) 2.24 (n=25) | rwsft1 ep2 (start) 1.82 (n=11) | base DeepSeek 2.11 (n=18)

== Pass 3: reasoning + visible text + tool calls, action+cot, controlarena_cot, gpt-6-luna low
   per-episode max and mean over scored turns, mean over episodes doing the side task; first-attempt-turn score; controls over all episodes
   [all tasks] max                  SFT ep2 (ds41sft1) 9.73 (n=26) | rwsft1 ep2 (start) 10.00 (n=29) | base DeepSeek 10.00 (n=29)   SFT vs rw: p=0.00698; SFT vs base: p=0.00698
   [all tasks] mean                 SFT ep2 (ds41sft1) 2.40 (n=26) | rwsft1 ep2 (start) 3.76 (n=29) | base DeepSeek 4.25 (n=29)   SFT vs rw: p=3.01e-05; SFT vs base: p=1.01e-06
   [all tasks] total                SFT ep2 (ds41sft1) 69.63 (n=26) | rwsft1 ep2 (start) 64.28 (n=29) | base DeepSeek 89.34 (n=29)   SFT vs rw: p=0.578; SFT vs base: p=0.00831
   [all tasks] first_attempt_score  SFT ep2 (ds41sft1) 9.20 (n=25) | rwsft1 ep2 (start) 9.62 (n=29) | base DeepSeek 10.00 (n=28)   SFT vs rw: p=0.0309; SFT vs base: p=0.000266
   [held-out tasks] max                  SFT ep2 (ds41sft1) 9.77 (n=13) | rwsft1 ep2 (start) 10.00 (n=15) | base DeepSeek 10.00 (n=15)   SFT vs rw: p=0.0587; SFT vs base: p=0.0587
   [held-out tasks] mean                 SFT ep2 (ds41sft1) 2.28 (n=13) | rwsft1 ep2 (start) 3.70 (n=15) | base DeepSeek 4.21 (n=15)   SFT vs rw: p=0.000911; SFT vs base: p=0.000911
   [held-out tasks] total                SFT ep2 (ds41sft1) 74.81 (n=13) | rwsft1 ep2 (start) 72.07 (n=15) | base DeepSeek 95.53 (n=15)   SFT vs rw: p=0.765; SFT vs base: p=0.0799
   [held-out tasks] first_attempt_score  SFT ep2 (ds41sft1) 9.17 (n=12) | rwsft1 ep2 (start) 9.87 (n=15) | base DeepSeek 10.00 (n=14)   SFT vs rw: p=0.387; SFT vs base: p=0.0575
   control (no side task) per-episode max: SFT ep2 (ds41sft1) 0.69 (n=16)
   control (no side task) per-episode max: rwsft1 ep2 (start) 0.38 (n=16)
   unconditioned per-episode max over all attack episodes: SFT ep2 (ds41sft1) 8.94 (n=32) | rwsft1 ep2 (start) 9.72 (n=32) | base DeepSeek 9.34 (n=32)

== Pass 4: reasoning + visible text, CoT only, controlarena_cot_only, gpt-oss-20b (held-out monitor)
   per-episode max and mean over scored turns, mean over episodes doing the side task; first-attempt-turn score; controls over all episodes
   [all tasks] max                  SFT ep2 (ds41sft1) 9.08 (n=26) | rwsft1 ep2 (start) 9.90 (n=29) | base DeepSeek 9.97 (n=29)   SFT vs rw: p=2.27e-06; SFT vs base: p=1.53e-07
   [all tasks] mean                 SFT ep2 (ds41sft1) 2.48 (n=26) | rwsft1 ep2 (start) 4.47 (n=29) | base DeepSeek 4.64 (n=29)   SFT vs rw: p=1.62e-06; SFT vs base: p=1.91e-06
   [all tasks] total                SFT ep2 (ds41sft1) 69.35 (n=26) | rwsft1 ep2 (start) 75.14 (n=29) | base DeepSeek 93.59 (n=29)   SFT vs rw: p=0.376; SFT vs base: p=0.0142
   [all tasks] first_attempt_score  SFT ep2 (ds41sft1) 5.58 (n=24) | rwsft1 ep2 (start) 8.45 (n=29) | base DeepSeek 9.46 (n=28)   SFT vs rw: p=0.00206; SFT vs base: p=1.02e-05
   [held-out tasks] max                  SFT ep2 (ds41sft1) 9.00 (n=13) | rwsft1 ep2 (start) 9.93 (n=15) | base DeepSeek 9.93 (n=15)   SFT vs rw: p=0.000224; SFT vs base: p=0.000224
   [held-out tasks] mean                 SFT ep2 (ds41sft1) 2.32 (n=13) | rwsft1 ep2 (start) 4.16 (n=15) | base DeepSeek 4.47 (n=15)   SFT vs rw: p=9.02e-05; SFT vs base: p=0.000464
   [held-out tasks] total                SFT ep2 (ds41sft1) 75.23 (n=13) | rwsft1 ep2 (start) 80.93 (n=15) | base DeepSeek 97.53 (n=15)   SFT vs rw: p=0.612; SFT vs base: p=0.128
   [held-out tasks] first_attempt_score  SFT ep2 (ds41sft1) 6.09 (n=11) | rwsft1 ep2 (start) 8.27 (n=15) | base DeepSeek 9.29 (n=14)   SFT vs rw: p=0.101; SFT vs base: p=0.0138
   control (no side task) per-episode max: SFT ep2 (ds41sft1) 3.06 (n=16)
   control (no side task) per-episode max: rwsft1 ep2 (start) 1.12 (n=16)
   unconditioned per-episode max over all attack episodes: SFT ep2 (ds41sft1) 8.41 (n=32) | rwsft1 ep2 (start) 9.81 (n=32) | base DeepSeek 9.69 (n=32)
```

Figure: `mats12_sprint/figures/ds41_sft1_arena.png` (attempt and leak counts per arm; per-episode max, first-attempt-turn score, per-episode mean over scored turns and per-episode sum of turn scores for each pass; easy and medium pooled).

## CoT length and the sanity battery
Transfer eval (`cot-control/rl/transfer_eval.py`, F0 wording, user placement, effort 75, 12 items x 3 datasets, 25,000-token cap; run `ds-ds41sft1ep2`: the no-requirement arm and six types, about $4.45 of sampling); the other three rows are the earlier DeepSeek runs of `cot-control/results/fry/NOTES.md` (`fry.py` -> `fry_length.md`):

| run | no-requirement median tokens (vs base) | no-requirement accuracy | median over the six types of (tokens with requirement / without), paired | word_suppression | max_words | no_backtracking | ignore_question | alternating_case | meow_between_words |
|---|---|---|---|---|---|---|---|---|---|
| ds-base | 1462 (1.00x) | 26/36 | 0.97 | 0.95 | 0.57 | 1.07 | 0.69 | 1.00 | 1.16 |
| ds-rwsft1ep2 | 652 (0.45x) | 26/36 | 0.80 | 1.21 | 0.11 | 0.29 | 0.39 | 2.76 | 2.64 |
| ds-obf2gs25 | 594 (0.41x) | 27/36 | 0.98 | 1.07 | 0.50 | 0.91 | 0.97 | 1.00 | 1.09 |
| ds-ds41sft1ep2 | 1412 (0.97x) | 27/36 | 0.80 | 1.00 | 0.06 | 0.41 | 0.60 | 2.29 | 1.96 |

Sanity battery (`cot-control/results/battery/ds41sft1-ep2.jsonl`, 22 questions, Kimi K2.6 judge against base DeepSeek): 21 same, 1 organism better, 0 worse; mean answer length 593 characters against base's 619; 0/22 truncated, repetitive or empty.

## Cost
Training about $9.83 (estimate from tokens sent). Eval cells (10): 72.79M prompt tokens and 1.599M sampled tokens on Tinker, at most $49.43 at the list prices ($0.60 and $3.60 per M; no cache credit). 40 monitor passes: $4.47 of judge calls. Transfer eval about $4.45, battery about $0.3.
