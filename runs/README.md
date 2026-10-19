# Training run diary

One markdown file per training run launched anywhere under `mats12_sprint/` (any project or subproject), written when the run is launched. Its Outcome section gets filled in when the run ends. Smoke tests count if they take optimizer steps; a batch of related smoke runs can share one file.

File name: `YYYY-MM-DD_<project>_<run-name>.md` (launch date), so `ls` lists runs in order.

## Entry format

```markdown
# <run name>

- Launched: YYYY-MM-DD
- Project: <path to the training code>, <git branch @ commit, "+ uncommitted" if dirty>
- Command: `<exact command>`
- Output: <results dir>, <checkpoint list file or paths>

## Task
What is being trained, on what data, toward what behavior, and why. A few sentences.

## Params
| param | value |
|---|---|
| base model | ... |
| ... | ... |

## Notes
Anything nonstandard: reward details, data choices, known risks, deviations from the previous run. Omit if nothing.

## Outcome
Filled in when the run ends: status (finished / stopped / crashed), steps completed, cost, headline metrics, eval results, pointers to analysis. Objective numbers only; no interpretation unless Ethan asked for a specific point to be noted.
```
