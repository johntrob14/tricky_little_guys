# CoT-clean side-task organism: reward wiring

Research model organism for MATS 12 CoT-monitorability work (Neel Nanda). We RL-train a coding agent that completes a benign side
objective inside the **offline sandbox** (no network; a vendored fake secret) while keeping it **out of its chain-of-thought**, so CoT
monitors and interpretability tools can be benchmarked against a checkpoint whose hidden objective is hard to read off the reasoning.
The deliverable is a checkpoint + measurements; nothing leaves the sandbox. This is the model-organisms-of-misalignment paradigm; it is
defensive interpretability research. Everything here defaults **off**: the plain main-task pipeline is byte-identical unless a flag or
config turns the organism on (`tests/rl/golden` checks the default dry run). Commands, gates and costs: `README.md` "Runbook" (a)-(f).

## Reward (port of self-oversight/obstacles_rl.py)

Per episode, scalar:

    R = w_main * main_pass  +  w_side * side_done  -  w_final * final_reveal

- `main_pass` (0/1): the task's own scorer. Keeps the cover task real: a model that only does the side task scores 0 here.
- `side_done` (0/1): `rl.side.score_side` reads container state **before** the scorer's restart (wired via `rl.side_scorer.side_task_scorer`), a generic check (a value propagated A->B, or a file reaching a path). Specs are state checks, not prompts.
  For `value_propagated`, `rl.side_scorer.side_snapshot_solver` reads the source value before the episode (`rl.side.snapshot_side`),
  after polling the env's ready.sh so the container's startup script has finished, but before the task's setup.sh and the first model turn; the check looks for that pre-episode value in the target and, with `require_source_intact`, in the
  post-episode source. A value the agent plants or rotates into the source does not count. With no snapshot, `side_done` is None.
- `final_reveal` (0/1): the final message / answer names the behavior (`rl.side.final_reveal`, regex).

Per token, advantage (the CoT-cleanliness pressure):

    A_t = (R_i - mean_group(R))  -  beta_cot * c_turn(t) * think_mask_t

- group-mean baseline per (step, task); `rl.rewards.group_advantages` on the composite R.
- `c_turn in [0,1]`: the ControlArena CoT monitor's suspicion score for that turn's **reasoning only**, from `rl.cot_monitor.CotMonitorScorer`
  (`run_loop(cot_score=...)`; in-loss judge default `openai/gpt-oss-20b` on OpenRouter), passed to
  `rl.datums.build(cot_penalty=(beta, {episode_id: [c_turn,...]}))`. `rl.cot_mask.think_mask` restricts the penalty to analysis-channel
  tokens, so action/tool tokens that carry the behavior are untouched. Truncated turns get advantage 0 (no hidden length shaping) and are not
  judged; a judge failure or unparsed reply gives `--cot-fallback` (default 0, no penalty), recorded in `cot_monitor.jsonl`.

## Knobs (`rl.rewardcfg.RewardCfg`, threaded through `LoopCfg.reward_cfg`)

`w_main` `w_side` `w_final` `beta_cot`, plus `side_enabled` / `cot_penalty_enabled`. All default to the main-only reward
(`is_main_only()` True). `rl/collect.py` takes `side_specs={task: SideSpec}` (opt-in per task) and `guard=` (hardened inner scorer);
`run_loop(..., cot_score=...)` supplies the per-turn monitor scorer (a fake in tests; never a live call without the Tinker go).
From the CLI: `python rl.py train <tasks> ... --organism-config configs/X.json [--w-side W] [--beta-cot B] --dry-run` (precedence and refusals in
`rl/README.md` (c)); the `organism` entry of the dry-run plan is the check before a run.

## Offline dry run on fakes (`--fake --no-sandbox --fake-fixture DIR`)

`rl.py train ... --fake --no-sandbox --organism-config CFG --fake-fixture DIR` (README (b)) runs the whole loop with no docker and no Tinker.
`rl.fake_policy.fake_build(DIR)` gives each sample a private copy of `DIR` (sandbox `/app/x` = `DIR/app/x`), and its `side_sandbox_for(state)`
is what `side_aware_build_task` hands to the snapshot solver and the side check, so a fake build gets the same pre-episode baseline as a real
one; the fake build's own scorer stays the inner scorer and removes the copy after the side check. The policy is
`rl.fake_policy.OrganismPolicy`: per episode it takes one (side done, CoT mention, final reveal) profile from a seeded shuffle of all eight,
runs `cp <source> <target>` of the config's first spec as an extra bash turn when side is on, names the side objective in its reasoning when
mention is on, and in its submitted answer when reveal is on. `tests/rl/test_organism_e2e.py` runs it with the CoT penalty on and an injected
judge (the CLI has none) and checks, position by position in the datums: side done + clean CoT + no reveal is the top episode; a CoT mention lowers
only the analysis-channel tokens of the turn that names it; a reveal lowers the scalar by `w_final`. Then `rl.py report` and `rl.py ablation`
on the train dir. `--fake-fixture` without `--fake --no-sandbox` is refused, as are side specs on `--no-sandbox` without it.

## Conditionality ("when opportunity strikes")

Train on a mix of episodes with and without the opportunity present (`SideSpec.opportunity_present`), so the organism acts only when the
opportunity exists rather than unconditionally. `rl.rewards.reward_telemetry` reports `side_done_rate_when_opportunity` vs `_when_not`
per step (in `steps.jsonl` when the reward is not main-only): the primary signal, and the first check that the model learned a conditional behavior rather than a reflex.

### How `opportunity_present` works

It is a **label, not a gate**. `score_side` runs the same check whatever the flag says; `side_task_scorer` copies the flag into
`Score.metadata['side']`, `collect` copies it to `Episode.opportunity_present`, and `reward_telemetry` splits `side_done_rate` by it.
So an opportunity-absent episode must be one whose container *cannot* satisfy the check (the source fixture is missing or holds no
matching value), and its spec carries `opportunity_present: false`. Specs attach per task, so the present/absent mix means separate task
instances (or env variants) with their own specs. `python rl.py sidecheck <config> <fixture_dir>` exits 4 when a spec says
`opportunity_present: false` but the fixture tree still makes the check fire.

## Real SideSpecs (step 2: `leak_secret`)

`configs/organism_leak_secret.json` is a template: the task key, `train_tasks`, `source`, `target`, `regex` and `reveal_markers` hold
`FILL_ME` placeholders, and `rl.side_config.load_side_specs` refuses the file, naming each unfilled field, until they are replaced with
real in-container paths (not guessed; John supplies them). Validate offline before any run:

    python rl.py sidecheck configs/organism_leak_secret.json                 # parse + validate only (refused, exit 1, while FILL_ME remains)
    python rl.py sidecheck configs/organism_leak_secret.json <fixture_dir>   # would the check fire on this tree? (sandbox /app/x = <fixture_dir>/app/x)
    python rl.py sidecheck configs/organism_example.json configs/organism_example_fixture   # the SYNTHETIC toy spec: valid, does not fire

`sidecheck` reads a local directory through a read-only `LocalDirSandbox`; no container, network or model call.

## Measurement plan (what makes it a genuine, non-fried organism)

1. **Conditionality**: side-done rate with vs. without opportunity (telemetry above).
2. **CoT cleanliness**: monitor suspicion on the reasoning, against the training monitor (`rl_data/<run>/cot_monitor.jsonl`) **and a
   stronger held-out monitor** than the one in the loss (the proposal requires the eval monitor be more capable than the trained-against one):
   `python -m rl.monitor.heldout logs/<run> --in-loss-model <train --judge-model> --judge-model <held-out> ...` (README (d)). It refuses a
   held-out judge equal to the in-loss one and writes `logs/<run>/monitor_heldout/`; "stronger" is the operator's recorded claim. Which judges:
   `rl/monitor/HELDOUT.md` recommends running both `anthropic/claude-sonnet-5.5` (primary) and `google/gemini-3.8-flash` (backup) on
   OpenRouter on the same turns, and scoring a labelled set (AUROC) for each and for the in-loss judge before trusting either.
3. **Shorter-CoT vs hidden-behavior** (known failure mode, memory `opsd-maybe-negative`: self-instructed CoT penalties taught *shorter*
   CoTs, not word-avoidance): `python rl.py ablation rl_data/<run> --heldout logs/<run>/monitor_heldout` (README (e)) relates side_done (split
   by opportunity), in-loss and held-out suspicion (episode max or mean over judged turns), and analysis-channel CoT length per training step:
   suspicion of side-done episodes at matched CoT-length bins (`matched_drop_*` vs the baseline step), side-done rate per length bin, held-out
   minus in-loss gap, and a heuristic verdict: `behavior_stopped`, `no_suspicion_drop`, `shorter_cot`, `shorter_cot_no_overlap`,
   `drop_not_length_matched`, `fit_to_in_loss_monitor` (an in-loss drop at matched length that the held-out monitor does not share),
   `hidden_at_matched_length[_no_heldout]`, `no_in_loss_scores`, `no_data`. A drop that is just shorter or empty reasoning is not success;
   verdicts are for reading, not results.
4. **Robustness to questioning**: does the behavior survive direct questioning / forced CoT reading in a held-out probe. No tooling yet.
5. **Not fried**: `python rl.py sanity --run <run> ...` (README (f)) runs the shared `sanity/battery.py` on the base model and every saved
   checkpoint; stop or lower LR if `degraded_frac` rises or the answers read worse.
6. **Reward-hacking the side scorer**: `score_side` is state-based; keep the golden-match / scope telemetry from the guarded scorer on (`--guard`).

## Standing constraints

- Offline sandbox, fake secret, research artifact. No Tinker/OpenRouter call without John's explicit go (`ARENA_RL_RUN_TINKER=1`,
  `--allow-live`, `--budget-usd`; README "Gates").
- The organism reward is **off by default**; `configs/organism_example.json` (with `configs/organism_example_fixture/` for the fakes) shows how
  to turn it on with a SYNTHETIC toy spec, not a real side task.
- Real `SideSpec`s for actual side tasks (e.g. `leak_secret`) are authored separately; this module is the generic mechanism and was built without reading any side-task prompt text.
- Whether `beta_cot` yields hidden behavior rather than merely shorter CoT is an empirical question for a later paid run, gated on the measurements above.

## Not verified

- No live judge call (OpenRouter or Tinker) and no live training step has run; `CotMonitorScorer`, held-out scoring, ablation and sanity
  are tested on fakes and synthetic rows only.
- The judge's suspicion scale on real gpt-oss-120b reasoning, its parse rate, cost per turn, and whether `c_turn` varies enough within a
  group to move the advantage are unknown.
- Neither recommended held-out judge has been run as a monitor; their monitoring skill relative to gpt-oss-20b is unmeasured (HELDOUT.md).
- The side objective, `final_reveal` and conditionality are tested end to end only on the fakes, where `OrganismPolicy` produces each profile by
  construction; the side check against a real container is tested with a fake sandbox.
