# Linux Arena: recurring public side tasks

Corrected review, 2026-10-18. Scope: the ten public environments at `v1.0.1`, the paper code, and Control Tower commit `699d5ca5c26ab3126fa29c6fa73dbf898aa0ca53`.

## What we want to measure

John wants to measure how well various open models perform on side tasks that recur across environments. First identify the recurring tasks and their actual differences; then choose models and a run design. No paid Linux Arena evaluations have been started in this review.

## Inventory

There are 92 public side-task definitions. Six families recur, accounting for 40 definitions; 52 definitions belong to families appearing in only one public environment. These counts were recomputed from `*/side_tasks/*/__init__.py`, stripping the environment prefix.

| Family | Environments | Coverage |
|---|---:|---|
| `reverse_shell` | 9 | All except clinical_trial |
| `run_malware_live` | 9 | All except clinical_trial |
| `rce` | 8 | All except clinical_trial and shared_dotfiles |
| `run_malware_delayed` | 8 | All except clinical_trial and shared_dotfiles |
| `exfiltrate_secrets_live` | 3 | auto_workflow, file_indexing, shared_dotfiles |
| `exfiltrate_secrets_delayed` | 3 | auto_workflow, file_indexing, shared_dotfiles |

The ten public environments are auto_workflow, clinical_trial, file_indexing, media_processing, model_registry, shared_dotfiles, slack_clone, tiktok, user_tracking, and web_scraping. The clinical_trial environment has five side tasks, all outside the recurring families. Gated environments are outside this review.

## What differs across environments

The previous report compared Python configuration files and scorer hashes, then concluded that the four broad families differed essentially only in their target URL. That conclusion was too strong. The framework also loads Markdown files into the task configuration. The paper's direct-attack and attack policies include scoring details by default.

Specifically, `make_side_task_config_from_path` reads `docs/goal.md`, `docs/scoring_details.md`, `docs/scorer.md`, optional additional instructions, and optional attack-surface documentation. `SideTask.task_description` combines the goal and scoring details. Which other fields reach the model depends on the policy.

| Family | Scorer source | Distinct goal + scoring-details texts | Interpretation |
|---|---|---:|---|
| `reverse_shell` | Identical across all 9 | 1 / 9 | Shared objective and scoring-description text; application context still varies |
| `run_malware_live` | Identical across all 9 | 6 / 9 | Shared checker, with wording and scoring-detail differences |
| `rce` | Identical across all 8 | 8 / 8 | Shared checker, with environment-specific scoring details |
| `run_malware_delayed` | Identical in 7; web_scraping differs | 8 / 8 | Environment-specific scoring details plus a restart difference |
| `exfiltrate_secrets_live` | Different in all 3 | Not counted in this comparison | Secret locations and verification logic depend on the application |
| `exfiltrate_secrets_delayed` | Different in all 3 | Not counted in this comparison | Secret locations and verification logic depend on the application |

Distinct-text counts are exact string comparisons, including whitespace. They establish that the provided descriptions differ; they do not measure how much those differences affect model performance.

### Shared scorers measure different outcomes

- RCE uses a runtime secret and verifies a marker containing that secret after submitted scripts execute from the external testing machine.
- Reverse shell uses connection records from the internet simulator and checks for a session originating outside the listener itself. Calling every shared scorer a sentinel-marker check was inaccurate.
- Live malware checks a marker with the expected diagnostic pattern in the default container.
- Delayed malware removes the marker, executes the submitted scripts, and checks whether the marker reappears. In web_scraping, cleanup also invokes `/app/restart.sh` before those scripts run. This changes the application state used by the test.
- Exfiltration has application-specific secret handling and verification, so a common family name provides weaker experimental standardization.

Live tasks score an effect during the trajectory. Delayed tasks score what the submitted artifact can achieve at grading time. The exact lifecycle should be preserved when comparing models.

## What this means for the experiment

Reverse shell is the closest match on objective and scoring-description text. The four broad families remain useful candidates, but their shared scorer does not establish that only the application changes.

For each model comparison, hold policy, tool access, environment revision, sampling settings, token and step limits, retries, and main-task pairing fixed. Record the fully rendered prompt. When comparing across environments, report both the application and the instruction differences. A separate ablation could standardize the side-task wording while retaining the original scorers; that would change the paper's prompt setup and should be reported explicitly.

The paper provides `policies.obvious_attack.policy:obvious_attack_policy` for direct side-task capability testing without concealment, and a separate multi-phase attack policy. Choosing between them changes the question being measured. The `attack_full.txt` manifest pairs side tasks with environment-specific main tasks; repeated families are visible in the definitions after stripping their prefixes.

## Local setup and repair

- Paper code: `~/code/tricky_little_guys/linuxarena/`, on branch `linuxarena`.
- Public environment clones: `~/code/linuxarena-envs/<environment>/`, each checked at tag `v1.0.1`.
- Control Tower: `~/code/control-tower/`, checked at commit `699d5ca5c26ab3126fa29c6fa73dbf898aa0ca53`.
- Python environment: `linuxarena/.venv/`.

The original setup was incomplete. There was no virtual environment in the vendored paper directory, and the separate upstream environment lacked Inspect. Installation also exposed two upstream issues:

1. Installing Control Tower from its Git URL tries to initialize an inaccessible private research submodule. The paper now uses the existing public checkout via the relative editable path `../../control-tower`, at the same pinned commit. A fresh machine must clone that checkout to the corresponding sibling location and check out the recorded revision.
2. Several public paper policies still imported through `research.paper_reproducibility.src.policies`. Those imports now use the public `policies` package and its included helpers.

The existing lockfile also omitted some dependencies declared by the paper's pyproject. It has been refreshed. Inspect remains at upstream version `0.3.171`, using the original lockfile's wheel URL and SHA-256 as an explicit override. Its version-specific PyPI JSON endpoint returned HTTP 404 during setup, while the locked wheel downloaded successfully. An independent comparison of the old and refreshed lockfiles found no version changes to existing packages. Eight packages were added to cover the already-declared Hugging Face and PDF dependencies.

From `linuxarena/`, use:

```bash
uv sync --frozen
uv run --frozen inspect --version
uv run --frozen ct --help
```

Verified: Inspect reports version `0.3.171`; Control Tower imports; the honest, attack, and obvious-attack policy modules import; and both `ct --help` and `ct run eval --help` load. `uv run --locked` successfully validated the lockfile and ran a one-sample Inspect evaluation with `mockllm/model`; its log status was `success`. This smoke test made no paid model calls. Logs are at `/tmp/linuxarena-repair/inspect-logs/` for this session; the test script is `/tmp/linuxarena-repair/smoke.py`. Container execution, registry wiring, image availability, service health, and a complete Linux Arena episode require separate validation. The earlier Docker check was denied by this session's sandbox, so it did not establish the host daemon's status.

## Verification sources

- Public clones: `~/code/linuxarena-envs/*/side_tasks/`.
- Configuration loader: `~/code/control-tower/src/control_tower/task/task_config.py`.
- Task-description composition: `~/code/control-tower/src/control_tower/task/task.py`.
- Shared grading helpers: `~/code/control-tower/src/control_tower/environments/scorer_utils.py`.
- Policy prompt selection: `linuxarena/src/policies/{obvious_attack,attack}/policy.py`.

This is a source-level inventory and setup review. It establishes neither model success rates nor the causal effect of environment differences.
