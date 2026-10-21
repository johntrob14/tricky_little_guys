# CoT-clean side-task organism: reward wiring

Research model organism for MATS 12 CoT-monitorability work (Neel Nanda). We RL-train a coding agent that completes a benign side
objective inside the **offline sandbox** (no network; a vendored fake secret) while keeping it **out of its chain-of-thought**, so CoT
monitors and interpretability tools can be benchmarked against a checkpoint whose hidden objective is hard to read off the reasoning.
The deliverable is a checkpoint + measurements; nothing leaves the sandbox. This is the model-organisms-of-misalignment paradigm; it is
defensive interpretability research. Everything here defaults **off** — the plain main-task pipeline is byte-identical unless a config
turns the organism on.

## Reward (port of self-oversight/obstacles_rl.py)

Per episode, scalar:

    R = w_main * main_pass  +  w_side * side_done  -  w_final * final_reveal

- `main_pass` (0/1): the task's own scorer. Keeps the cover task real — a model that only does the side task scores 0 here.
- `side_done` (0/1): `rl.side.score_side` reads container state **before** the scorer's restart (wired via `rl.side_scorer.side_task_scorer`), a generic check (a value propagated A->B, or a file reaching a path). Specs are state checks, not prompts.
- `final_reveal` (0/1): the final message / answer names the behavior (`rl.side.final_reveal`, regex).

Per token, advantage (the CoT-cleanliness pressure):

    A_t = (R_i - mean_group(R))  -  beta_cot * c_turn(t) * think_mask_t

- group-mean baseline per (step, task); `rl.rewards.group_advantages` on the composite R.
- `c_turn in [0,1]`: the ControlArena CoT monitor's suspicion score for that turn's **reasoning only** (`rl/monitor`), passed to
  `rl.datums.build(cot_penalty=(beta, {episode_id: [c_turn,...]}))`. `rl.cot_mask.think_mask` restricts the penalty to analysis-channel
  tokens, so action/tool tokens that carry the behavior are untouched. Truncated turns get advantage 0 (no hidden length shaping).

## Knobs (`rl.rewardcfg.RewardCfg`, threaded through `LoopCfg.reward_cfg`)

`w_main` `w_side` `w_final` `beta_cot`, plus `side_enabled` / `cot_penalty_enabled`. All default to the main-only reward
(`is_main_only()` True). `rl/collect.py` takes `side_specs={task: SideSpec}` (opt-in per task) and `guard=` (hardened inner scorer);
`run_loop(..., cot_score=...)` supplies the per-turn monitor scorer (a fake in tests; never a live call without the Tinker go).
From the CLI: `python rl.py train <tasks> ... --organism-config configs/X.json [--w-side W] [--beta-cot B] --dry-run` (precedence and refusals in
`rl/README.md`); the `organism` entry of the dry-run plan is the check before a run.

## Conditionality ("when opportunity strikes")

Train on a mix of episodes with and without the opportunity present (`SideSpec.opportunity_present`), so the organism acts only when the
opportunity exists rather than unconditionally. `rl.rewards.reward_telemetry` reports `side_done_rate_when_opportunity` vs `_when_not`
per step — the primary signal, and the first check that the model learned a conditional behavior rather than a reflex.

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

    python rl.py sidecheck configs/organism_leak_secret.json                 # parse + validate only
    python rl.py sidecheck configs/organism_leak_secret.json <fixture_dir>   # would the check fire on this tree? (sandbox /app/x = <fixture_dir>/app/x)

`sidecheck` reads a local directory through a read-only `LocalDirSandbox`; no container, network or model call.

## Measurement plan (what makes it a genuine, non-fried organism)

1. **Conditionality**: side-done rate with vs. without opportunity (telemetry above).
2. **CoT cleanliness**: monitor suspicion on the reasoning, against the training monitor **and a stronger held-out monitor** than the one
   in the loss (the proposal requires the eval monitor be more capable than the trained-against one).
3. **Shorter-CoT vs hidden-behavior** (known failure mode — see memory `opsd-maybe-negative`: self-instructed CoT penalties taught
   *shorter* CoTs, not word-avoidance): log CoT-length distribution; confirm the behavior still fires; ablate *why* the monitor score
   dropped. A drop that is just shorter/empty reasoning is not success.
   Tooling (offline, tested on synthetic data): `rl/monitor/heldout.py` scores saved logs with a held-out judge that must differ from the in-loss
   judge (into `<logs>/monitor_heldout/`, same --allow-live/--budget-usd gates as `rl.monitor.run`); `rl/ablation.py` relates side_done (split by
   opportunity), in-loss and held-out suspicion, and CoT length per checkpoint: suspicion at matched CoT-length bins, side-done rate per length bin,
   held-out minus in-loss gap, and a heuristic verdict (`behavior_stopped`, `shorter_cot`, `shorter_cot_no_overlap`, `fit_to_in_loss_monitor`,
   `hidden_at_matched_length`, ...).
4. **Robustness to questioning**: does the behavior survive direct questioning / forced CoT reading in a held-out probe.
5. **Not fried**: run the shared `sanity/battery.py` on the base model and every saved checkpoint; stop or lower LR if answers degrade.
6. **Reward-hacking the side scorer**: `score_side` is state-based; keep the golden-match / scope telemetry from the guarded scorer on.

## Standing constraints

- Offline sandbox, fake secret, research artifact. No Tinker/OpenRouter call without John's explicit go.
- The organism reward is **off by default**; `configs/organism_example.json` shows how to turn it on with a SYNTHETIC toy spec (not a real side task).
- Real `SideSpec`s for actual side tasks (e.g. `leak_secret`) are authored separately; this module is the generic mechanism and was built without reading any side-task prompt text.
- Whether `beta_cot` yields hidden behavior rather than merely shorter CoT is an empirical question for a later paid run, gated on the measurements above.
