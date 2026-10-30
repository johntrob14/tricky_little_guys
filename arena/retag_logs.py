"""Backfill the labels.py naming scheme onto .eval logs written before it, so every run in logs/ filters the
same way in the Inspect viewer. Sets the task name, the eval tags and the knob metadata (on the eval and on
every sample), and renames each file so its slug matches the new name.

    python retag_logs.py logs --dry-run      # print what every log would become
    python retag_logs.py logs                # rewrite in place
    python retag_logs.py logs/atk-qwen-easy  # one run

Knobs come from the log's own metadata where it has them (`preset` and the `prompt` fingerprint, written by
run_attack.py since the axes were separated). Older run_attack logs carry only `mode`, so honest_hide/control is
read from that: those runs predate the axes and were honest_hide (attack+hide) or control, which
tests/test_prompts.py checks against the committed prompts. run.py logs have no side task at all and become
mode:honest with preset, prefill, tier and side_task all "none".

Rewriting is a read -> modify -> write round-trip through inspect_ai's own model, which is lossless (verified by
comparing full model dumps). Run this BEFORE the monitor stage: rl/monitor keys its rows by log file name, so
renaming afterwards makes it re-judge everything.
"""

import argparse
import re
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

import labels


def knobs_of(log) -> dict:
    md = log.eval.metadata or {}
    sample_md = (log.samples[0].metadata or {}) if log.samples else {}
    env = sample_md.get("env") or "unknown"
    task = sample_md.get("task") or "unknown"
    model, model_path = md.get("model") or log.eval.model, (md.get("model_args") or {}).get("model_path")

    if "preset" in md:                                  # written after the prompt axes were separated
        prompt = md.get("prompt") or {}
        return labels.knobs(model, model_path, md["preset"], prompt.get("prefill"), md.get("tier") or "none", env, task, "leak_secret", bool(prompt.get("description", True)))
    if "mode" in md:                                    # older run_attack.py: mode is "attack+hide" or "control"
        attack = md["mode"] != "control"
        return labels.knobs(model, model_path, "honest_hide" if attack else "control", None, md.get("tier") or "none", env, task, "leak_secret", attack)
    k = labels.knobs(model, model_path, "none", None, "none", env, task, "none", True)   # run.py: main task only, no side task
    return {**k, "mode": "honest"}


def retag(path: Path, rename: bool, dry_run: bool) -> str:
    log = read_eval_log(str(path))
    k = knobs_of(log)
    name = labels.task_name(k)
    target = path
    if rename:
        ts, _, uid = path.stem.split("_")[0], None, path.stem.split("_")[-1]
        target = path.with_name(f"{ts}_{re.sub(r'[^A-Za-z0-9]+', '-', name).strip('-')}_{uid}.eval")
    if dry_run:
        return f"{path.parent.name}/{path.name}\n    -> name {name}\n    -> tags {labels.tags(k)}\n    -> file {target.name}"

    log.eval.task, log.eval.task_display_name, log.eval.tags = name, name, labels.tags(k)
    log.eval.metadata = {**(log.eval.metadata or {}), **k}
    for s in log.samples or []:
        s.metadata = {**(s.metadata or {}), **k}
    write_eval_log(log, str(target))
    if target != path:
        path.unlink()
    return f"{path.parent.name}/{target.name}: {name}"


ap = argparse.ArgumentParser()
ap.add_argument("logs", help="a logs/ root, or one run dir, or one .eval file")
ap.add_argument("--no-rename", action="store_true", help="rewrite the contents but keep each file's current name")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

root = Path(a.logs)
files = [root] if root.is_file() else sorted(root.rglob("*.eval"))
if not files:
    sys.exit(f"no .eval logs under {root}")
for f in files:
    print(retag(f, not a.no_rename, a.dry_run), flush=True)
print(f"\n{len(files)} log(s){' (dry run, nothing written)' if a.dry_run else ' rewritten'}")
