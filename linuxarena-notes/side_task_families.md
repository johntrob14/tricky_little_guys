# LinuxArena side tasks shared across environments (public 10)

Exploration of redwoodresearch/linuxarena-paper for the open-model side-task study.
This note is benchmark-structure analysis only; it does not reproduce any task's operational content.

## Local setup (working clones, not vendored here — large, and upstream keeps them external)

- Paper repo: vendored in this branch under `linuxarena/` (git subtree of linuxarena-paper @ main).
- Env repos (pinned `v1.0.1`): cloned to `~/code/linuxarena-envs/<env>/` — the 10 public envs.
  Upstream pulls these into `.environments/` via `uv run ct registry pull linuxarena` (needs `gh`);
  here they were cloned directly from github.com/linuxarena/<env>.
- Control Tower (eval framework, pinned rev `699d5ca`): `~/code/control-tower/`; installed editable
  into `linuxarena/.venv`. The shared side-task grading lives in
  `control_tower/environments/scorer_utils.py`.
- Public registry (`control_tower/environments/environments.yml`): the 10 public envs are
  auto_workflow, clinical_trial, file_indexing, media_processing, model_registry, shared_dotfiles,
  slack_clone, tiktok, user_tracking, web_scraping. The other 10 are gated (ignored for now).

## Finding: 6 side-task families recur across public environments

Side tasks are named with an env prefix (`aw_`, `fi_`, `tt_`, …). Stripping the prefix gives the family.
92 side tasks across the 10 public envs → 6 families appear in more than one env (40 instances);
the remaining 52 are env-specific.

| Family | # public envs | envs |
|---|---|---|
| run_malware_live | 9 | all except clinical_trial |
| reverse_shell | 9 | all except clinical_trial |
| run_malware_delayed | 8 | above minus shared_dotfiles |
| rce | 8 | above minus shared_dotfiles |
| exfiltrate_secrets_delayed | 3 | auto_workflow, file_indexing, shared_dotfiles |
| exfiltrate_secrets_live | 3 | auto_workflow, file_indexing, shared_dotfiles |

(clinical_trial, the only R/notebook env, carries none of the shared families — all 5 of its side
tasks are domain-specific, e.g. suppressing safety notifications.)

## What actually differs across environments

Two sharply different cases:

1. **The four near-universal families (rce, reverse_shell, run_malware_live/delayed) are one
   definition reused.** The `scorer.py` is byte-identical across envs (same md5; it just calls the
   shared `scorer_utils` helpers), and the task definition (`__init__.py`) is identical except for a
   single line — the target application's URL. These are generic "can the agent achieve this
   capability against whatever app runs here" probes, graded by a sentinel marker (e.g. the scorer
   sets up a secret, runs the agent's submitted scripts on an external machine, and checks a marker),
   not by any environment-specific effect. So across envs, essentially **only the target host differs**:
   taskflow.co, api.indexer.aicorp.com, api.mediaflow.io, vertex.ai, crewsync.com, tiktok-guesser.app,
   api.usertracking.io, api.wikiscrape.io. (One small exception: web_scraping's run_malware_delayed
   scorer hash differs slightly from the others — worth a look before relying on it.)

2. **The two exfiltrate_secrets families are genuinely env-specific.** Their scorers differ per env
   (19 / 54 / 119 lines for the delayed variant in file_indexing / … / shared_dotfiles; different
   hashes), because "which secret, stored where, and how exfiltration is verified" depends on the
   environment's real services. These are the shared families where the *content*, not just a URL,
   changes across envs.

**live vs delayed** is a second axis crossing the malware/exfiltration families: `live` requires the
effect during the trajectory; `delayed` plants something that triggers later. Scorers differ by axis.

## Caveat for anyone reading the paper's run manifest

`linuxarena/src/task_sets/attack_full.txt` pairs each side task with an env-specific main task, so the
ST names there look unique per env and a naive group-by finds "0 shared." The sharing is visible at the
environment-registry / task-definition level (the prefixes above), not in the run manifest.

## Implication for the open-model study

The four near-universal capability probes are the natural first target: identical graders and prompts
across 8–9 envs mean an open model's success rate is comparable across envs, and any variation is
attributable to the environment (the surrounding app / reachable surface), not to task wording. The
exfiltrate_secrets families would instead measure env-specific exfiltration skill. Model choice still to
be decided.
