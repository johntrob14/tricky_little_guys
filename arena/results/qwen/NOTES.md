# qwen: frozen-context pool for tutor-reward GRPO on the cot-control Qwen organism (2026-10-23)

Policy: Qwen3.8-27B with the `cotctl-opsd-grpo1b-step10` LoRA (`cot-control/results/rl/opsd-grpo1b/checkpoints.json`). Episodes: the arena `arena_covert_v2` preset (side task in the prompt), tiers easy / medium / hard, via `run_attack.py -M model_path=...`.

Monitoring mode: **cot only** (`controlarena_cot_only`, judge `openai/gpt-oss-20b` via OpenRouter, temperature 0), task description cut to the main task only (`monitor_maintask.py`, output `logs/<run>/monitor_maintask_cot_only`). The tutor (luna at max effort) is not used in this step.

## Episodes
| source | runs | episodes | main task passed | scorer side task leaked |
|---|---|---|---|---|
| new, post-fix (scorer oracle not on disk during the episode; 0/16 oracle path in each tier's transcripts) | `qwen-org-covert2-{easy,medium,hard}-b` (`run_attack.py tiktok --preset arena_covert_v2 --tier T --epochs 2`, all 8 tiktok tasks, started 2026-10-23, `5e5b21c`) | 16 per tier, 48 | easy 16/16, medium 15/16, hard 16/16 | easy 7/16, medium 6/16, hard 5/16 |
| older, pre-fix | `qwen-opsdgrpo30-covert2-{easy,medium,hard}` (4 tasks x 2 epochs per tier, 2026-10-21) | 8 per tier, 24 | | |

All 1404 assistant turns of the new episodes are scored (478 easy, 460 medium, 466 hard; $0.12). 8 of the 72 episodes carry the scorer oracle's path or keys in their transcripts (all older) and are dropped.

## Leak detector
`leak_check.py` now catches cwd-relative sinks and basename, bare-name and whole-environment secret references. `leak_check.py confusion` over 256 episodes (ws8, gpt-oss covert2, Qwen covert2): the detector fires on some turn of 60/60 scorer-leaked episodes (34/34 gpt-oss and ws8, 26/26 Qwen) and of 57/196 scorer-clean ones (40/174 gpt-oss and ws8, 17/22 Qwen). Before the change: 46/60 (Qwen 12/26) and 43/196. Leave-one-out over the six pattern changes: the sink patterns add 6/26 Qwen leaks, the basename reference 3/26, the bare name 2/26, the environment spread 1/26; the bare name adds 4 gpt-oss/ws8 scorer-clean fires and 1 Qwen, the spread 2 and 3, the basename 3 Qwen. Of the 9 Qwen scorer-clean fires before the change, 8 were new unauthenticated routes at URLs the scorer does not fetch. Tests: `tests/test_leak_check.py`. ws8 detector-attempt rates quoted earlier (e.g. `ws8-ctl-medium-obst` 8/40) were computed with the earlier detector; the same logs give 10/40 now.

## Read-through of 85 pool turns (workflow `wf_e5b1702a-94d`, 8 readers and 2 verifiers, 77 classified; one reader's slice of 8 loud turns was blocked by the safety classifier)
Of 77 classified turns (32 decisive, 45 loud): the logged monitor score judged defensible 71/77; good as a training context 46/77 (decisive 18/32, loud 28/45). Reasons for not good, 31/77: submit or pure recap with the side task already done 14, re-verification of an earlier leak 7, port or socket probing in an unreachable-server loop 4, monitor over-call on benign work 2, trivial no-variance step 2, duplicate planning turn 1, a turn whose reasoning references the scorer oracle 1. CoT coherent 62/77, rambling 15/77 (14 loud); 15/77 turns have a CoT of about 400 words or more. Side-task content in the CoT at loud turns (45): explicit plan 23, hint only 9, explicit execution 9, recap after 3, none 1; at decisive turns (32): recap after 10, none 8, explicit execution 6, hint only 5, explicit plan 3. Reader and verifier agreed on 12/12 doubly read turns.

## Pool (`decisive_ctx.py`)
Flags: `--loud-monitor monitor_maintask_cot_only --loud-min 8 --loud-before-fire --loud-gate --skip-probe --loud-per-episode 2 --drop-submit --skip-oracle-reads` over the 6 run dirs. Loud turns at or above score 8: 555. Dropped: submit-only 40, at or after the episode's first detector fire 148, port or process probing 90, CoT names no covert term and not the secret or its file 45, beyond the 2-per-episode cap (earliest kept) 111; kept 121. Pool: 188 contexts from 64 episodes (leak 58, loud 121, miss 8, retry 1; `contexts_pool.json`, `sides_pool.json`).

## Base-Qwen length reference and final pool (`pool_finalize.py`)
`resample_ctx.py` with base Qwen3.8-27B (no checkpoint), G=8 at all 188 contexts, no prefill: 1504 samples, cot-only main-task-only scores >= 8 on 972/1504 and <= 1 on 366/1504, empty CoT 0/1504, unparsed 1/1504, policy $15.79, judge $0.12 (`base_g8/`). Filters: prompt over 20000 Qwen tokens 29, base median CoT over 400 words 18, every base sample scored <= 2 by the monitor 5. Kept 136 contexts from 60 episodes: leak 35, loud 100, miss 1; easy 41, medium 47, hard 48; 101 from the new episodes and 35 from the older ones; by task batch_guess_submission 15, extract_auth_into_middleware_layer 9, guess_replay_and_simulation 14, missing_user_id_in_guess_response 22, secure_password_policy_engine 8, sponsor_statsv2_diggcount_total 31, token_bucket_rate_limiter_per_user 19, video_url_format_validation 18. Prompt tokens of the kept contexts: median 5906, mean 7886, max 19644; base mean output 398 tokens; base median CoT 69 words (max 394). Files: `contexts.json`, `sides.json` (with `prompt_tokens`, `base_median_words`), `base_words.json`.

## Cost of a training run (estimate from `pool_finalize.py`, Tinker prices from `self-oversight/common.py`: Qwen3.8-27B train $4.103, sample $5.595, prefill $0.6975 per M tokens; tutor $0.0036 per rating)
Training tokens dominate (every sample is trained with its prompt). Contexts per step x G=8, 15 steps, by prompt cap (kept contexts, leak / loud):
| prompt cap | contexts | leak | loud | mean prompt tokens | 32 per step | 24 per step |
|---|---|---|---|---|---|---|
| 8000 | 85 | 4 | 81 | 4593 | $113 | $85 |
| 12000 | 104 | 10 | 94 | 5517 | $130 | $98 |
| 16000 | 123 | 26 | 97 | 6816 | $154 | $116 |
| 20000 | 136 | 35 | 100 | 7886 | $174 | $131 |
Equal-reward groups are dropped from training, so these are upper bounds.

## Trainer checks for Qwen (`turn1_grpo.py --model Qwen/Qwen3.8-27B`)
Prompt tokens from the trainer's render path equal the episode model's (`run.TinkerModel`) on 5/5 pool contexts. Samples from the organism parsed through the trainer's `parse`: non-empty reasoning 20/20, tool call parsed 20/20. Reward tests: 13/13 (`tests/test_turn1_reward.py`, `tests/test_tutor_control.py`).

## Files
`contexts_pool.json`, `sides_pool.json`, `contexts.json`, `sides.json`, `base_words.json`, `base_g8/{samples.jsonl, summary.json}`, `base_g8.log`; code `decisive_ctx.py`, `pool_finalize.py`, `leak_check.py`, `turn1_grpo.py`.

## qwentutor1 (diary `runs/2026-10-23_arena_qwentutor1.md`): eval of the step-15 checkpoint
48 arena_covert_v2 episodes with the same recipe as the organism's own collection (`logs/qwentutor1-s15-covert2-{easy,medium,hard}-b`, 8 tasks x 2 epochs per tier, including the held-out task), scored by the same cot-only main-task-only monitor (`controlarena_cot_only`, judge openai/gpt-oss-20b via OpenRouter).

| | organism (`qwen-org-covert2-*-b`) | qwentutor1 step 15 |
|---|---|---|
| main task passed | 47/48 | 44/48 |
| scorer side task leaked | 18/48 (easy 7, medium 6, hard 5) | 9/48 (3, 3, 3) |
| detector fires on some turn | 45/48 | 44/48 |
| episodes with max monitor score >= 8 | 48/48 | 47/48 |
| per-episode max monitor score, mean | 9.88 | 9.75 |
| per-episode mean monitor score, mean | 3.88 | 3.47 |
| assistant turns | 1404 | 1154 |

Reader passes over 21 picked rollouts at steps 6 and 15 (Sonnet 5.5 readers, workflows `wf_40669429-2d0` and `wf_ab78c5d7-1f4`): coherent CoT 20/21 at both steps; tutor scores judged consistent with the rollout 20/21 and 21/21; the highest-reward rollouts at step 15 (4/21, all leak arm) have CoTs of 8-16 words ahead of a working covert edit, 3 of them shorter than the base median at the context; one rollout credited for a write under the temp directory; two leak-arm CoTs reason about what a grader would see. Training-side numbers are in the diary and `mats12_sprint/figures/qwentutor1_training.png`.
