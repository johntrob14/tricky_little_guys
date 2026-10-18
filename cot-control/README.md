# cot-control

CoT-Control (Chen et al., [YuehHanChen/CoTControl](https://github.com/YuehHanChen/CoTControl)) experiments: can a model obey instructions
about the style or content of its own reasoning trace?

| path | contents |
| --- | --- |
| `latent/` | Squashed subtree of [johntrob14/cot-control-latent](https://github.com/johntrob14/cot-control-latent) `main` at `2a31131`. The `cotswitch` package (prompts, item draws, CoT extraction, graders, judge prompts, analysis tables) and the SWITCH latent-thinker experiments (Qwen3-8B + SWITCH LoRA vs base Qwen3-8B). Its raw pod results (`results_local/`, 2.1 GB, git-ignored upstream) are not included; the report is in `latent/notes/report-2026-10-14.md`. |
| `tinker-runner/` | Tinker runner and judge backend for Neel Nanda's Astra variant (108 items: 9 constraints x 3 datasets x 4). History from the local `cot-control-deepseek` repo. Its README has every run, the exact commands, verification notes and caveats. |
| `results/tinker-runs/` | Raw generations, grades, judgments and analysis tables for every Tinker run (copied from `~/code/results/cot-control-deepseek` on zeus). |

## Tinker results so far (Astra, pooled compliance excluding meta-discussion)

| model | sampling | "reasoning stage" wording | "thinking" wording | run dirs |
| --- | --- | --- | --- | --- |
| DeepSeek V4.1 Flash | greedy, effort 75 | 0/108 | 0/108 | `run1`, `run2` |
| DeepSeek V4.1 Flash | T 1.0, top_p 0.95, effort 75 | not run | 0/108 | `ds-t1-run2` |
| GLM-5.3 | T 1.0, top_p 0.95, effort high | 12/108 | 17/108 | `glm53-run1`, `glm53-run2` |
| Nemotron-3-Ultra | T 1.0, top_p 0.95, full thinking | 1/108 (pipeline says 2; one judge false positive) | 6/108 | `nemo-ultra-run1`, `nemo-ultra-run2` |
| Nemotron-3-Ultra | T 1.0, top_p 0.95, medium effort | stopped at 67/108, ungraded | 1/108 | `nemo-ultra-med-run1`, `nemo-ultra-med-run2` |

Trace length differs a lot between models (GLM thinks briefly, Nemotron at length), so these rates mix control with verbosity.

## Running from the monorepo

`tinker-runner/tinker_cotcontrol.py` now finds `cotswitch` in `../latent` (set `COTSWITCH_ROOT` to override). Checked after vendoring:
regrading `glm53-run2` with `latent/` reproduces the stored grades exactly, and it builds the same 108 prompts. The analysis commands in
`tinker-runner/README.md` still `cd ~/code/cot-control-switch-opt`; from here, run them in `cot-control/latent` instead:

    cd cot-control/latent && PYTHONPATH=. python -m cotswitch.cli grade   --run-dir <run dir>
    python ../tinker-runner/tinker_cotcontrol.py judge --run-dir <run dir>
    cd cot-control/latent && PYTHONPATH=. python -m cotswitch.cli analyze --run-dir <run dir>

The runner needs `tinker`, `transformers`, `jinja2`, `numpy`, `pandas`, `pyyaml`, `scipy`, `matplotlib` and `tqdm`, and the Tinker key in
`~/.config/sauce/tinker.env`.
