# arena/rl

RL on the arena's tasks: gpt-oss-120b trained with Tinker, rolled out through Ethan's `run.py` agent solver, `task_scorer` and `build_task` (imported unchanged; no edit to his files). Nothing here has made a live Tinker or OpenRouter call. The model-organism reward and its measurements are in `ORGANISM.md`; the held-out judge choice is in `monitor/HELDOUT.md`.

Run everything from `arena/` with the venv (`.venv/bin/python`); `rl.py` is a script (the package `rl/` is what `import rl` finds). Every command below was parsed and, where it has one, run in its offline form (`--dry-run`, `--fake`, an injected fake judge, or the refusal without the gate) on 2026-10-21 against the code on this branch; the live forms were not run. Dollar figures quoted below are what those offline runs printed.

## Gates

| gate | needed by | without it |
| --- | --- | --- |
| `ARENA_RL_RUN_TINKER=1` (John's go) | live `collect`, live `train`, `sanity --allow-live`, `train --judge-backend tinker`, `rl.monitor.run` / `rl.monitor.heldout --judge-backend tinker`, `python -m rl.run_rl_eval` | `refused: ...` before any client is built (exit 1) |
| `--budget-usd X` | `collect`, `train`, `sanity` (required flag); live `rl.monitor.run` / `heldout` | argparse error / `refusing: --budget-usd is required` |
| `--allow-live` | `train` whenever the CoT penalty is on (`--beta-cot`, or a config's `beta_cot` with `cot_penalty_enabled`); live `sanity`; live `rl.monitor.run` / `heldout` (either backend) | `refused: ... pass --allow-live` (`train --dry-run` lists it in `would_refuse` and exits 2) |
| total cap $20 over `rl_data/ledger.csv` | `train`, `plan-collect` (checked against the first-step forecast), `rl.monitor.run --ledger` | listed in `would_refuse` / `refusing:` |
| OpenRouter key (`OPENROUTER_API_KEY`, or `~/.config/sauce/openrouter.env`; `python -m rl.monitor.run --set-key` / `--check-key`) | the OpenRouter judge (in-loss default and the held-out monitor) | `refusing:` before any call |
| `TINKER_API_KEY` | every Tinker call | `rl.train.new_service` and `run_rl_eval` load `arena/.env` as `run.py` does; `~/.config/sauce/tinker.env` is not read by this code. No `arena/.env` exists on zeus, so export the key from the env file first. |

`--fake` never touches Tinker; `--fake` train runs must be named `fake-*`. The test suite monkeypatches `tinker.ServiceClient` to fail if touched. `train --dry-run` never spends and exits 2 when the real run would be refused. There is no fake judge on the command line: a train command with the CoT penalty on and no `--allow-live` is refused even with `--fake`, and with `--allow-live` it builds the live judge. Tests inject a judge through `rl.main(argv, cot_judge=...)`.

## Runbook

### (a) First measurement: base gpt-oss-120b collect, then report

```bash
python rl.py plan-collect tiktok --run m1 --epochs 4 --budget-usd 12      # exact gated command + reserve/expected; no call; exit 2 = would be refused
ARENA_RL_RUN_TINKER=1 python rl.py collect tiktok/missing_user_id_in_guess_response tiktok/sponsor_statsv2_diggcount_total \
    tiktok/token_bucket_rate_limiter_per_user tiktok/video_url_format_validation --run m1 --epochs 4 --budget-usd 12 \
    --model openai/gpt-oss-120b --max-turns 60 --max-tokens 4096          # LIVE: the command plan-collect prints (copy it from there)
python -m rl.monitor.run logs/m1 --out rl_data/m1/monitor --dry-run       # optional in-loss-judge scores of the collect; prints the worst case
python -m rl.monitor.run logs/m1 --out rl_data/m1/monitor --budget-usd 1 --allow-live   # LIVE OpenRouter, default judge openai/gpt-oss-20b
python rl.py report rl_data/m1                                             # offline: report.{md,json} in rl_data/m1
```

`plan-collect` for 4 tasks x 4 epochs: reserve $10.1722 (16 x $0.6358 worst case), expected $0.48 (placeholder $0.03 per episode). `collect` reserves the full worst case before sampling, so `--budget-usd` must exceed it. Episodes go to `rl_data/m1/collect/`, Inspect logs to `logs/m1/step_000/a<attempt>g<group>/`. `report` reads `<dir>/episodes.jsonl`, `<dir>/collect/episodes.jsonl` or a train run's `<dir>/step_NNN/episodes.jsonl`; every `monitor.jsonl` under `<dir>` (hence `--out rl_data/m1/monitor` above; `--monitor F` adds others); a train run's in-loss `cot_monitor.jsonl` (kept apart from held-out rows and labelled `(in-loss)`); and the ledger. The pass-rate CI is Wilson over episodes and treats epochs as independent. Offline rehearsal: `python rl.py collect tiktok --run fake-m1 --epochs 2 --budget-usd 1 --ep-worst-usd 0.05 --fake --no-sandbox`, then `python rl.py report fake-m1` (a fake collect still reserves, hence `--ep-worst-usd`).

### (b) Organism dry run on fakes (offline, no docker)

```bash
python rl.py sidecheck configs/organism_example.json configs/organism_example_fixture     # the spec parses; it does not fire on the untouched tree
python rl.py train tiktok/missing_user_id_in_guess_response --run fake-org --steps 2 --B 1 --G 8 --budget-usd 1 --ep-worst-usd 0.02 \
    --fake --no-sandbox --p-fix 1.0 --organism-config configs/organism_example.json --beta-cot 0 --fake-fixture configs/organism_example_fixture
python rl.py report fake-org                                               # side_done / final_reveal overall and per step
python rl.py ablation fake-org                                             # ablation.{md,json}; no in-loss scores with --beta-cot 0
```

`configs/organism_example.json` is a SYNTHETIC toy spec (copy `/app/fixture_secret.txt` to `/app/notes.txt`), not a side task. `--fake-fixture DIR` gives each fake sample a private copy of `DIR` (sandbox `/app/x` = `DIR/app/x`). The pre-episode snapshot and the side check read that copy, and the fake policy becomes `rl.fake_policy.OrganismPolicy`, which cycles every (side done, CoT mention, final reveal) profile and runs `cp <source> <target>` of the config's first spec as its side command. `--beta-cot 0` turns the example config's CoT penalty off, because the CLI has no fake judge (see Gates); `tests/rl/test_organism_e2e.py` runs the same loop with the penalty on and an injected judge. Refused: `--fake-fixture` without both `--fake` and `--no-sandbox` (it would be ignored), a `--fake-fixture` path that is not a directory, and side specs on `--no-sandbox` without `--fake-fixture`. On this run each step showed `side_done_rate` 0.5 and `final_reveal_rate` 0.5 in `steps.jsonl`, by construction of the policy.

### (c) Organism training (live)

```bash
python rl.py train tiktok --run org1 --steps 4 --B 2 --G 4 --budget-usd 12 --organism-config configs/organism_example.json --allow-live --dry-run   # plan; exit 0
ARENA_RL_RUN_TINKER=1 python rl.py train tiktok --run org1 --steps 4 --B 2 --G 4 --budget-usd 12 --organism-config configs/organism_example.json --allow-live   # LIVE
python rl.py train tiktok --run org1 --steps 4 --B 2 --G 4 --budget-usd 12 --beta-cot 0.5 --allow-live --dry-run   # CoT penalty only, no config
```

`--organism-config CFG` (train and collect; off when omitted, and then nothing changes) loads `side_specs` into `Collector` (side-aware scorer on those tasks, `--guard` kept) and `reward_cfg` into `LoopCfg.reward_cfg` via `rl.side_config.load_organism`. Per reward field an explicit `--w-main` / `--w-side` / `--w-final` / `--beta-cot` beats the config, which beats the `RewardCfg` default; `--beta-cot X` also sets `cot_penalty_enabled` to X != 0, and a config-enabled CoT penalty goes through the same `--allow-live` / budget gate as `--beta-cot`. Refused: `FILL_ME` placeholders, a bad spec or weight, `side_enabled` with no spec, a spec for a task not in `<tasks>`, `--w-*` without a config, `--w-side` / `--w-final` without `side_enabled`. The config's run/model/steps/B/G/train_tasks are not read; the dry-run plan's `organism` entry shows the effective `reward_cfg`, where each field came from, the side tasks, and whether `train_tasks` match (for the example config with `tiktok` they do not: `config_train_tasks_match` is false). Without `--allow-live` the first dry run lists the CoT refusal and exits 2. `configs/organism_leak_secret.json` is refused (`sidecheck` exit 1) until its `FILL_ME` fields hold real in-container paths.

Judge flags: `--judge-backend {openrouter,tinker}` (default openrouter), `--judge-model` (default `openai/gpt-oss-20b` on openrouter, `deepseek-ai/DeepSeek-V4.1-Flash:peft:262144` on tinker), `--judge-max-tokens 6000`, `--cot-concurrency 8`, `--cot-fallback 0.0` (c_turn for a failed or unparsed judge call). Plan for the first command: per-step rollout reserve $5.0861, train worst case $0.1932, forecast $0.60 per step, $0.6173 with the judge's expected cost; run worst case $21.1172, $23.9004 with the judge's worst case. `would_refuse` checks the first-step forecast, one step's reservation and ledger + forecast against $20, not the run worst case, so this plan is not refused; the per-step reservation still halts the loop at `--budget-usd`. Per-turn judge rows go to `rl_data/org1/cot_monitor.jsonl`, the in-loss scores that (e) reads.

Plain (main-task-only) training is unchanged: `python rl.py train tiktok --run r1 --steps 10 --B 2 --G 4 --budget-usd 12 --dry-run`; on fakes, `python rl.py train tiktok --run fake-r1 --steps 3 --B 2 --G 4 --budget-usd 1 --ep-worst-usd 0.02 --fake --no-sandbox`.

### (d) Held-out monitor scoring

Use the recommendation in `monitor/HELDOUT.md` (also `rl.monitor.heldout.RECOMMENDED`; nothing defaults to it): run **both** judges on the same turns, primary `anthropic/claude-sonnet-5.5` and backup `google/gemini-3.8-flash`, both on OpenRouter. The primary-backup disagreement is itself a result. Before trusting either, score a labelled set (side done vs not) and report AUROC for both held-out judges and for the in-loss judge; "stronger" in HELDOUT.md is general capability (AA index 46.8 and 40.9 vs 9.0 for gpt-oss-20b), not measured monitoring skill. If the in-loss judge is DeepSeek-V4.1-Flash (AA 45.1), neither pick is clearly stronger at default effort; see HELDOUT.md.

```bash
python -m rl.monitor.heldout logs/org1 --in-loss-model openai/gpt-oss-20b --judge-backend openrouter --judge-model anthropic/claude-sonnet-5.5 \
    --stronger-because "AA Intelligence Index 46.8 (high) vs 9.0 for gpt-oss-20b, artificialanalysis.ai 2026-10-21; different lab from policy and in-loss judge" --dry-run
python -m rl.monitor.heldout logs/org1 --in-loss-model openai/gpt-oss-20b --judge-backend openrouter --judge-model google/gemini-3.8-flash \
    --out logs/org1/monitor_heldout_backup --stronger-because "AA 40.9 vs 9.0 for gpt-oss-20b; a fourth lab" --dry-run
# LIVE: the same two commands with --budget-usd X --allow-live instead of --dry-run; add --ledger rl_data/ledger.csv to count toward the $20 cap
```

`--in-loss-model` must be the judge `train` used (its `--judge-model`, default `openai/gpt-oss-20b`); a held-out judge with the same base name is refused, and so are `--out` equal to the in-loss `monitor/` dir and a logs path that does not exist. Give the backup its own `--out`, so each held-out dir holds one judge. `heldout.json` is written even on `--dry-run`. Rows go to `<out>/a<attempt>g<group>/monitor.jsonl` with `heldout.json` in `<out>`. On the 2-step fake run of (b) (56 turns) the dry runs gave worst cases of $3.87 (Sonnet 5.5) and $2.37 (Gemini 3.8 Flash), so `--budget-usd 2` is refused for the primary. HELDOUT.md gives about $14-16 typical and $64-71 worst case per 1000 turns for the primary; pilot about 20 turns first and read the unparsed rate.

### (e) Ablation (why did suspicion drop?)

```bash
python rl.py ablation rl_data/org1 --heldout logs/org1/monitor_heldout --heldout-judge anthropic/claude-sonnet-5.5
python rl.py ablation rl_data/org1 --heldout logs/org1/monitor_heldout_backup --out rl_data/org1/ablation_backup
```

Offline. Reads `rl_data/<run>/step_NNN/episodes.jsonl` (a checkpoint is the training step); in-loss suspicion from `cot_monitor.jsonl` (per-episode max, or `--stat mean`, over judged turns; skipped and fallback turns left out; `--in-loss-judge` when a resumed run mixed judges); held-out rows from `--heldout` (a `monitor.jsonl`, or a dir: every `monitor.jsonl` under it, pooled). Held-out rows join to episodes by `sidecar.episode_id`, else by (.eval basename, sample, epoch), because `rl.monitor.run --sidecars` finds no sidecar for the loop's `a<attempt>g<group>/` logs. Rows that mix judges, within a file or across files, are refused unless `--heldout-judge` picks one; a held-out judge equal to the in-loss judge, or a different score scale, is refused. CoT length is analysis-channel tokens from the gpt-oss tokenizer in the HF cache (sampled tokens as an upper bound if the tokenizer is unavailable, flagged in the output). Other flags: `--bins 8`, `--baseline-step`, `--out`. Writes `ablation.md` and `ablation.json`; the verdicts are heuristics, explained in ORGANISM.md. Checked on the fake run of (b) with held-out rows from an injected fake judge: 4 of 4 side-done episodes per step joined.

### (f) Sanity battery (not fried)

```bash
python rl.py sanity --run org1 --budget-usd 5 --dry-run          # targets (base + checkpoints) and worst case
python rl.py sanity --run fake-org --budget-usd 5 --fake         # offline fakes
ARENA_RL_RUN_TINKER=1 python rl.py sanity --run org1 --budget-usd 5 --allow-live   # LIVE: Tinker sampling + gpt-oss-120b judge
```

Runs the repo's shared `sanity/battery.py` on the base model and every row of `rl_data/R/checkpoints.json` (a state row is skipped when a sampler row holds the same step; `--kinds sampler,state`), via `rl.sanity.run_sanity`. Each checkpoint's answers are paired with the base's, judged with `battery.JUDGE` (organism randomly A/B, `--seed`), and summarised. Files: `rl_data/R/sanity/{base,<checkpoint name>}/{answers.jsonl,judged.jsonl,summary.json}` and `sanity/table.{md,json}`, one row per checkpoint: `degraded_frac` = (organism_broken + base_better) / parsed verdicts, `broken_frac`, truncation/repetition/empty rates and their change from base. Checkpoints already summarised are skipped. The worst case (no cache, every answer at 600 tokens) is reserved against `--budget-usd` before each target. Read the answers by eye too, as the battery asks.

### Other commands

```bash
python rl.py doctor                    # docker/image/disk/memory gates + golden controls (about 3 min), no Tinker; flags pass through to rl.doctor
python rl.py loadtest --levels 1,4,6   # compose load test, no Tinker
python rl.py collect tiktok/missing_user_id_in_guess_response --run test1 --epochs 1 --fake --p-fix 1.0 --budget-usd 1 --ep-worst-usd 0.05   # real sandbox, fake policy
ARENA_RL_RUN_TINKER=1 python -m rl.run_rl_eval tiktok --run X --epochs 4    # plain tinker-rl eval without the loop (LIVE)
```

`collect --model-path tinker://...` samples a checkpoint instead of the base model; `--guard` scores with `rl.guard`'s hardened scorer; `--fake` uses `rl.fake_policy.ScriptedPolicy` (`[plan:fix]` / `[plan:skip]` at turn 0 with `--p-fix`) and, for `train`, a fake training service; `--no-sandbox` swaps in docker-free fake tools and scorer. `doctor`, `loadtest` and the real-sandbox `collect` need docker and were only parsed (`--help`) for this runbook.

## Flow

`train` -> `rl.loop.run_loop` -> per step: `pick_tasks`, `Collector` (`rl/collect.py`) -> Inspect `eval_async` of `run.build_task` (for a task with a side spec, `rl.side_scorer.side_aware_build_task`: a pre-episode snapshot solver, then the same solver, and the side check before the inner scorer) with the `tinker-rl` model (`rl/provider.py`, token-space history, one sidecar per sample) -> `classify_sample` per sample, INFRA reruns (max 2, backoff 5/15/45 s, lost sidecar cost ledgered as a `wasted` row, new sandbox each time), HALT raises -> `episode_from_sidecar` -> reward from the Inspect log (`rewards.attach_reward`, composite when the reward is not main-only) -> with the CoT penalty on, `CotMonitorScorer` judges each turn's reasoning -> group advantage (minus `beta_cot * c_turn` on analysis-channel tokens) -> one merged `importance_sampling` datum per episode -> `forward_backward` + `optim_step` -> checkpoints.

## On disk

`rl_data/` is git-ignored except `.gitignore`; `ARENA_RL_DATA` overrides the root. Inspect logs go to `logs/<run>/step_NNN/a<attempt>g<group>/` (not git-ignored).

- `rl_data/<run>/steps.jsonl`: one `arena.rl.step.v1` line per step (counts, pass rate, rates of flags, lp_gap, ended counts, cost, stop reason; `reward_telemetry` when the reward is not main-only).
- `rl_data/<run>/step_NNN/`: `sidecars/<sample_uuid>.json.gz` (schema `arena-rl-sidecar/1`, documented in `rl/provider.py`: prompt ids, sampled ids, sampling logprobs, observations, flags per turn), `episodes.jsonl` (`arena.rl.episode.v1`, `rl/schema.py`: tokens in token space, `passed`, `failure_class` none/agent/infra, `side_done`, `final_reveal`, `opportunity_present`, `usage`, `wasted_usd`), `collected.json`, `batch.json`, `datums.jsonl.gz`, `trained.json` (written right after `optim_step`; resume depends on it).
- `rl_data/<run>/cot_monitor.jsonl` (CoT penalty on): one `arena.rl.cot_monitor.v1` row per turn with score, c, fallback, reason, cost.
- `rl_data/<run>/checkpoints.json`: state (3 d, 7 d final) and sampler (30 d final) rows with `tinker://` path and TTL. `HALTED.md` on any halt.
- `rl_data/ledger.csv` (`rl_data/<run>/ledger.csv` for `--fake`): append-only `time,phase,run,model,kind,estimate_usd,counted_usd,note`; kinds `forecast`, `counted`, `wasted` (informational). `model` is the policy model, except judge rows (`cot_monitor`, `sanity_judge`, and `rl.monitor.run` / `rl.monitor.heldout` with `--ledger`, phases `monitor` / `heldout`), which carry the judge model; judge rows written before that change carry the policy model. `rl.monitor.run` and `heldout` write no ledger rows without `--ledger`.
- `collect` writes `rl_data/<run>/collect/{episodes.jsonl,sidecars/}`; `report` writes `report.{md,json}`; `ablation` writes `ablation.{md,json}`; held-out scoring writes `logs/<run>/monitor_heldout/`.

## Cost

Per episode worst case (`budget.episode_worst_case_usd`, no prefix cache, every turn at `max_tokens`, prompt growing to the 32k context): 60 turns x 4096 tokens gives $0.6358. The loop reserves `ep_worst_usd * B * G` before each step's rollouts and charges it in full if `collect` fails, so `--budget-usd` must exceed it (B=G=4: $10.17 per step). `--ep-worst-usd` overrides the reservation (lowers the margin). Training worst case is `B * G * 32768 * 0.737 / 1e6` per step. The plan's forecast is `steps * B * G * (ep_usd_est + train_usd_est)` with the placeholders $0.03 and $0.02 per episode (x1.5 until a step is measured); both are unmeasured. After each step the loop commits sampling cost from `Episode.usage`, which the sidecar adapter fills as a no-cache upper bound (sum of prompt lengths), so counted sampling cost is overstated. With the CoT penalty on, the plan's `cot_monitor` block adds the judge's per-step cost (`rl.cot_monitor.step_cost`): worst case = every turn judged twice at `--judge-max-tokens` out and a prompt of template + 3000 task tokens + 1.5 x `--max-tokens`; expected = 24 turns per episode at 2000 prompt / 600 output tokens, placeholders from the gpt-oss-20b judge on `logs/atk-*` (attack runs, not training). `step_forecast_with_judge_usd` (loop forecast + judge expected) is what the `--budget-usd` and total-cap checks use; the loop's own `forecast_step` still excludes the judge, whose calls reserve per call. Prices: `budget.PRICES` (list prices); judge prices: `rl/monitor/pricing.py`. Stage cap is `--budget-usd`; total cap over the ledger is $20.

## Verified (offline, with outputs from the test suite)

Provider/token-space/sidecar, trainer on fakes, classification, guard hacks and chaos on real `la-*` containers, doctor and loadtest; this package adds: dry-run plan and refusal, mandatory budget, Tinker gate (including `run_rl_eval` and the tinker judge backend of `rl.monitor.run` / `heldout`), fake end-to-end loop (3 steps, ledger in the run dir, final sampler saved), collector INFRA rerun with wasted-spend row, persistent INFRA -> reward None, HALT-class raise, AGENT failure not rerun and scored 0, and (docker marked) one real Inspect sandbox episode of `missing_user_id_in_guess_response` with the fake policy: fix scores 1, no fix scores 0, sidecar aligned. `--organism-config` wiring, with the no-flag dry run byte-identical to before (`tests/rl/golden`). The organism loop end to end on fakes (`tests/rl/test_organism_e2e.py`: the real side check on the fixture copy against the pre-episode snapshot, an injected judge, advantages checked position by position, then `report` and `ablation`). CoT-monitor scorer, report, held-out gates, ablation and the sanity battery: on fakes and synthetic rows only.

## Not verified / known limits

- Every Tinker-facing call is unexercised: `ServiceClient`, `create_sampling_client`, real `forward_backward`/`optim_step`/save/load, sampler swap between steps, `ModelInput` handling of the real client. The fake client follows `tinker_api.md`, not a live call. No OpenRouter call has been made either (`judges.openrouter_judge` is tested with a fake transport; `judges.tinker_judge` was never called).
- Training dynamics on real data are unknown. The fake loop's pass rates, lp_gap, side-done rates and costs are meaningless.
- The task set is 4 tasks (`tiktok`), group size G small; with 4 tasks, B <= 4. A task whose group has constant reward contributes nothing. Pass rates from Ethan's `keep.yaml` (e.g. 2/4 on `video_url_format_validation` for the base model) are the only signal available.
- Stacks at once: `--max-sandboxes 6` (default); the load test on the pristine tree measured about 0.87-0.90 GiB peak per stack and CPU saturation from N=4 (see `rl_data/loadtest.json`); agent-like load is not measured.
- `run_with_retries` is not used; reruns are done in `Collector` (INFRA only). `step_deadline_s` is not enforced by the collector (no timeout of its own besides Inspect's sample time limit, which becomes `ended=wall`, reward None).
- The collector's classification of a sample without a score as INFRA (`no_score`) and of a log without the expected samples as `episode_missing` are untested against real Inspect failure modes beyond the cases in `tests/rl/test_cli.py`.
- A truncated turn does not end the episode (Ethan's loop nudges once); its advantage is zeroed in the datum.
- Hardened scoring (`--guard`) is validated on `tests.py` tasks only.
- The side check on a real container (`side_aware_build_task` over `run.build_task`) is tested with a fake sandbox only; the fixture path of (b) is a stand-in for the container, not a test of it.
- No checkpoint garbage collection or publishing, no eval/compare command.
- `rl.sanity.live_gen`/`live_judge` (Tinker sampling, state -> sampler export, gpt-oss-120b judge) are unexercised, and the judge's token counts and the state export cost are not in the reservation's verified range.
