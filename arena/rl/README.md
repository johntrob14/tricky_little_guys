# arena/rl

RL on the arena's tasks: gpt-oss-120b trained with Tinker, rolled out through Ethan's `run.py` agent solver, `task_scorer` and `build_task` (imported unchanged; no edit to his files). Nothing here has made a live Tinker call.

## Run

From `arena/`, with the venv (`.venv/bin/python`); `rl.py` is a script (the package `rl/` is what `import rl` finds).

```bash
python rl.py doctor                    # docker/image/disk/memory gates + golden controls (about 3 min), no Tinker
python rl.py loadtest --levels 1,4,6   # compose load test, no Tinker
python rl.py collect tiktok/missing_user_id_in_guess_response --run test1 --epochs 1 --fake --p-fix 1.0 --budget-usd 1 --ep-worst-usd 0.05   # real sandbox, fake policy, phase=test
python rl.py train tiktok --run r1 --steps 10 --B 2 --G 4 --budget-usd 12 --dry-run   # plan + forecast, never spends (exit 2 = would be refused)
python rl.py train tiktok --run fake-r1 --steps 3 --B 2 --G 4 --budget-usd 1 --ep-worst-usd 0.02 --fake --no-sandbox   # whole loop on fakes
python rl.py train tiktok --run r1 --steps 10 --B 2 --G 4 --budget-usd 12             # LIVE: refused unless ARENA_RL_RUN_TINKER=1
python -m rl.run_rl_eval tiktok --run X --epochs 4    # plain tinker-rl eval without the loop (also live)
python rl.py plan-collect tiktok --run m1 --epochs 4 --budget-usd 12   # exact gated collect command + reserve/expected cost; no call (exit 2 = would be refused)
python rl.py report rl_data/m1                         # offline: pass rate + Wilson CI, RL signal, side/reveal, monitor c_turn, CoT lengths, failures, cost -> report.{md,json}
```

`--budget-usd` is mandatory for `collect` and `train`. `--guard` scores with `rl.guard`'s hardened scorer. `--fake` uses `rl.fake_policy.ScriptedPolicy` (a scripted bash/bash/submit policy, `[plan:fix]` or `[plan:skip]` chosen at turn 0 with `--p-fix`) and, for `train`, a fake training service; `--no-sandbox` swaps in docker-free fake tools and scorer. `--fake` train runs must be named `fake-*`.

## The Tinker gate

No command here calls Tinker unless `ARENA_RL_RUN_TINKER=1` is set, which needs John's go. Live `collect` and `train` exit with a refusal otherwise; the test suite monkeypatches `tinker.ServiceClient` to fail if it is touched. `~/.config/sauce/tinker.env` is not read by any of this code; `rl.train.new_service` loads `arena/.env` as `run.py` does.

## Flow

`train` -> `rl.loop.run_loop` -> per step: `pick_tasks`, `Collector` (`rl/collect.py`) -> Inspect `eval_async` of `run.build_task` with the `tinker-rl` model (`rl/provider.py`, token-space history, one sidecar per sample) -> `classify_sample` per sample, INFRA reruns (max 2, backoff 5/15/45 s, lost sidecar cost ledgered as a `wasted` row, new sandbox each time), HALT raises -> `episode_from_sidecar` -> reward from the Inspect log (`rewards.attach_reward`) -> group advantage -> one merged `importance_sampling` datum per episode -> `forward_backward` + `optim_step` -> checkpoints.

## On disk

`rl_data/` is git-ignored except `.gitignore`; `ARENA_RL_DATA` overrides the root. Inspect logs go to `logs/<run>/step_NNN/a<attempt>g<group>/`.

- `rl_data/<run>/steps.jsonl`: one `arena.rl.step.v1` line per step (counts, pass rate, rates of flags, lp_gap, ended counts, cost, stop reason).
- `rl_data/<run>/step_NNN/`: `sidecars/<sample_uuid>.json.gz` (schema `arena-rl-sidecar/1`, documented in `rl/provider.py`: prompt ids, sampled ids, sampling logprobs, observations, flags per turn), `episodes.jsonl` (`arena.rl.episode.v1`, `rl/schema.py`: tokens in token space, `passed`, `failure_class` none/agent/infra, `usage`, `wasted_usd`), `collected.json`, `batch.json`, `datums.jsonl.gz`, `trained.json` (written right after `optim_step`; resume depends on it).
- `rl_data/<run>/checkpoints.json`: state (3 d, 7 d final) and sampler (30 d final) rows with `tinker://` path and TTL. `HALTED.md` on any halt.
- `rl_data/ledger.csv` (`rl_data/<run>/ledger.csv` for `--fake`): append-only `time,phase,run,model,kind,estimate_usd,counted_usd,note`; kinds `forecast`, `counted`, `wasted` (informational).
- `collect` writes `rl_data/<run>/collect/{episodes.jsonl,sidecars/}`.

## Cost

Per episode worst case (`budget.episode_worst_case_usd`, no prefix cache, every turn at `max_tokens`, prompt growing to the 32k context): 60 turns x 4096 tokens gives $0.6358. The loop reserves `ep_worst_usd * B * G` before each step's rollouts and charges it in full if `collect` fails, so `--budget-usd` must exceed it (B=G=4: $10.17 per step). `--ep-worst-usd` overrides the reservation (lowers the margin). Training worst case is `B * G * 32768 * 0.737 / 1e6` per step. The plan's forecast is `steps * B * G * (ep_usd_est + train_usd_est)` with the placeholders $0.03 and $0.02 per episode (x1.5 until a step is measured); both are unmeasured. After each step the loop commits sampling cost from `Episode.usage`, which the sidecar adapter fills as a no-cache upper bound (sum of prompt lengths), so counted sampling cost is overstated. Prices: `budget.PRICES` (list prices). Stage cap is `--budget-usd`; total cap over the ledger is $20.

## Verified (offline, with outputs from the test suite)

Provider/token-space/sidecar, trainer on fakes, classification, guard hacks and chaos on real `la-*` containers, doctor and loadtest; this package adds: dry-run plan and refusal, mandatory budget, Tinker gate, fake end-to-end loop (3 steps, ledger in the run dir, final sampler saved), collector INFRA rerun with wasted-spend row, persistent INFRA -> reward None, HALT-class raise, AGENT failure not rerun and scored 0, and (docker marked) one real Inspect sandbox episode of `missing_user_id_in_guess_response` with the fake policy: fix scores 1, no fix scores 0, sidecar aligned.

## Not verified / known limits

- Every Tinker-facing call is unexercised: `ServiceClient`, `create_sampling_client`, real `forward_backward`/`optim_step`/save/load, sampler swap between steps, `ModelInput` handling of the real client. The fake client follows `tinker_api.md`, not a live call.
- Training dynamics on real data are unknown. The fake loop's pass rates, lp_gap and costs are meaningless.
- The task set is 4 tasks (`tiktok`), group size G small; with 4 tasks, B <= 4. A task whose group has constant reward contributes nothing. Pass rates from Ethan's `keep.yaml` (e.g. 2/4 on `video_url_format_validation` for the base model) are the only signal available.
- Stacks at once: `--max-sandboxes 6` (default); the load test on the pristine tree measured about 0.87-0.90 GiB peak per stack and CPU saturation from N=4 (see `rl_data/loadtest.json`); agent-like load is not measured.
- `run_with_retries` is not used; reruns are done in `Collector` (INFRA only). `step_deadline_s` is not enforced by the collector (no timeout of its own besides Inspect's sample time limit, which becomes `ended=wall`, reward None).
- The collector's classification of a sample without a score as INFRA (`no_score`) and of a log without the expected samples as `episode_missing` are untested against real Inspect failure modes beyond the cases in `tests/rl/test_cli.py`.
- A truncated turn does not end the episode (Ethan's loop nudges once); its advantage is zeroed in the datum.
- Hardened scoring (`--guard`) is validated on `tests.py` tasks only.
- No checkpoint garbage collection or publishing, no eval/compare command, no sanity battery.
