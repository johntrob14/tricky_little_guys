"""Re-score saved screen rollouts with the current detect.py and print the Phase 0 table.

    uv run python rescore.py                      # every results/screen/<model>/<condition>.jsonl
    uv run python rescore.py --models gpt-oss-20b --envs smolagents swe-agent

The screen saves each rollout's diff, so detector changes can be applied after the fact without
re-sampling. The denial judge labels are not recomputed (they were only requested for rollouts the
detector flagged at run time), so a rollout newly flagged here shows denial "n/a".
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import RESULTS
from detect import detect
from envs import load_env

CONDS = ("unprompted", "nudged", "pressure", "instructed")


def rescore(model_dir: Path, envs_filter: list[str] | None) -> dict:
    envs, tasks = {}, {}
    out: dict = {}
    for cond in CONDS:
        p = model_dir / f"{cond}.jsonl"
        if not p.exists():
            continue
        agg = defaultdict(lambda: defaultdict(int))
        flagged = []
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if envs_filter and r["env"] not in envs_filter:
                continue
            a = agg[r["cls"]]
            a["rollouts"] += 1
            a["cot_monitoring"] += r["cot_monitoring"]
            if not r["applied"]:
                continue
            a["applied"] += 1
            a["task_passed"] += bool(r["task_passed"])
            env = envs.setdefault(r["env"], load_env(r["env"]))
            key = (r["env"], r["task_id"])
            if key not in tasks:
                tasks[key] = json.loads((Path("envs") / r["env"] / "tasks" / f"{r['task_id']}.json").read_text())
            d = detect(r["diff"], tasks[key]["reference_diff"], sites=env.sites, repo=env.repo)
            a["masked"] += bool(d.masked_families)
            if d.action:
                a["action"] += 1
                flagged.append({"env": r["env"], "task": r["task_id"], "families": sorted(d.families()),
                                "denial": r["denial"] or "n/a", "hits": [h.reason + " | " + h.line.strip()[:90] for h in d.hits]})
        out[cond] = {"by_class": {k: dict(v) for k, v in agg.items()}, "flagged": flagged}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--show-hits", action="store_true")
    a = ap.parse_args()
    root = RESULTS / "screen"
    models = a.models or sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.endswith("-pilot"))
    print(f"{'model':<24} {'condition':<11} {'opp applied':>11} {'action':>6} {'masked':>6} {'task ok':>7} {'CoT':>5}  near-miss act  benign act")
    for m in models:
        res = rescore(root / m, a.envs)
        for cond, r in res.items():
            o = r["by_class"].get("opportunity", {})
            nm = r["by_class"].get("near_miss", {})
            bn = r["by_class"].get("benign", {})
            print(f"{m:<24} {cond:<11} {o.get('applied', 0):>11} {o.get('action', 0):>6} {o.get('masked', 0):>6} "
                  f"{o.get('task_passed', 0):>7} {o.get('cot_monitoring', 0):>3}/{o.get('rollouts', 0):<3} "
                  f"{nm.get('action', 0):>6}/{nm.get('applied', 0):<4} {bn.get('action', 0):>6}/{bn.get('applied', 0)}")
            if a.show_hits:
                for f in r["flagged"]:
                    print(f"    {f['env']}/{f['task']} {f['families']} denial={f['denial']}")
                    for h in f["hits"]:
                        print(f"      {h}")


if __name__ == "__main__":
    main()
