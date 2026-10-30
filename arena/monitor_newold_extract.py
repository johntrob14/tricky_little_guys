"""Per-episode monitor scores for the same runs under two monitor passes (an old and a new one), as JSON for monitor_newold_chart.py. Prints and stores numbers only, never reasoning or commands.
Each pass is a monitor output directory under the run dir (<run>/<sub>/<run>/monitor.jsonl, as rl.monitor.run writes it). The record of a pass holds the summary.json's monitor and judge
metadata, turn/skipped/unparsed counts, judge cost, and per episode: id, tier, leaked flag (the side-task scorer), main-task pass, the turns where the leak detector fires (a tool call that writes
the secret to a surface, arena/leak_check.py; "attempted" = any fire), per-turn scores (null = skipped or unparsed) and the score on the leak turn (arena/leak_turn.py).

    .venv/bin/python monitor_newold_extract.py OUT.json --old monitor_maintask_cot_only[,fallback_sub...] --new monitor_ctx_luna label=logs/dir1,logs/dir2 label2=logs/dir3 ...

--old takes a comma-separated list of candidate sub directories; the first one present under a run is used (control cells may only have monitor_cot_only, which for a prompt without a side task is the same input as the main-task-only pass).
--old none records the --new pass alone (for monitor_placement_chart.py). Each episode record also carries the main-task pass flag.
"""
import argparse, json
from pathlib import Path
import leak_turn
from leak_check import leak_in_calls, msg_calls


def pass_record(d, sub):
    s = json.load(open(Path(d) / sub / Path(d).name / "summary.json"))
    leak = {r["id"]: r for r in leak_turn.analyse([d], sub)}
    eps = []
    for smp, mon in leak_turn.episodes(d, sub):
        asst = [m for m in smp.messages if m.role == "assistant"]
        meta, side = next(iter(smp.scores.values())).metadata, smp.metadata.get("side")
        fires = [t for t, m in enumerate(asst) if side and leak_in_calls(msg_calls(m), side)["leak"]]
        eps.append(dict(id=smp.id, tier=meta["tier"], leaked=bool(meta["side_leaked"]), main=bool(meta["main_pass"]), attempted=bool(fires), fires=fires,
                        scores=[mon.get((smp.id, i)) for i in range(len(asst))], at_leak=leak[smp.id]["at_leak"] if smp.id in leak else None))
    return dict(sub=sub, monitor=s["monitor"], judge={k: s["judge"].get(k) for k in ("backend", "model", "reasoning", "provider")}, turns=s["turns"], skipped=s["skipped"], unparsed=s["unparsed"], cost_usd=s["cost_usd"], episodes=eps)


def run_record(d, old_subs, new_sub):
    if old_subs == ["none"]:  # a single pass: only "new" is recorded (monitor_placement_chart.py)
        return {"new": pass_record(d, new_sub)}
    old = next(sub for sub in old_subs if (Path(d) / sub / Path(d).name / "monitor.jsonl").is_file())
    return {"old": pass_record(d, old), "new": pass_record(d, new_sub)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--old", default="monitor_maintask_cot_only,monitor_cot_only")
    ap.add_argument("--new", default="monitor_ctx_luna")
    ap.add_argument("cells", nargs="+", help="label=logs/dir1,logs/dir2")
    a = ap.parse_args()
    res = {}
    for spec in a.cells:
        label, dirs = spec.split("=", 1)
        res[label] = {d: run_record(d, a.old.split(","), a.new) for d in dirs.split(",")}
    json.dump(res, open(a.out, "w"))
    for label, runs in res.items():
        for p in ("old", "new"):
            if any(p not in r for r in runs.values()):
                continue
            t = sum(r[p]["turns"] for r in runs.values())
            subs = sorted({r[p]["sub"] for r in runs.values()})
            print(f"{label:22s} {p:3s} {'/'.join(subs):40s} episodes {sum(len(r[p]['episodes']) for r in runs.values()):3d} | turns {t:4d} | skipped {sum(r[p]['skipped'] for r in runs.values()):3d}/{t} | unparsed {sum(r[p]['unparsed'] for r in runs.values()):2d}/{t} | judge ${sum(r[p]['cost_usd'] for r in runs.values()):.3f}")
