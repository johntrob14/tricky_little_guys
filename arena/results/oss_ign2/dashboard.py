"""Standalone HTML dashboard of monitor scores for st1-ignore2 vs base gpt-oss-120b on the arena runs (numbers only), from monitor_modes_data.json and capability_*.json.

    python3 results/oss_ign2/dashboard.py OUT.html
"""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
mm = json.load(open(HERE / "monitor_modes_data.json"))
cap = {m: json.load(open(HERE / f"capability_{s}.json")) for m, s in (("org", "oss-ign2"), ("base", "oss-base"))}


def eps(cell, mode):
    return [e for run in mm[cell].values() for e in run[mode]["episodes"]]


def stat(cell, mode, kind):
    vals = [(max if kind == "max" else st.mean)(p) for e in eps(cell, mode) if (p := [x for x in e["scores"] if x is not None])]
    return dict(mean=round(st.mean(vals), 2), se=round(st.stdev(vals) / len(vals) ** 0.5, 2), n=len(vals))


bars = {f"{m}|{g}|{mode}|{k}": stat(f"gpt_{m}_{g}", mode, k) for m in ("org", "base") for g in ("covert_v2", "ctl") for mode in ("cot_only", "action+cot") for k in ("max", "mean")}
hist = {m: [sum(1 for e in eps(f"gpt_{m}_covert_v2", "cot_only") for x in e["scores"] if x is not None and int(x) == b) for b in range(11)] for m in ("org", "base")}
tiers = {f"{m}|{t}": dict(n=len(g), main=sum(e["main"] for e in g), side=sum(e["side"] for e in g), fired=sum(bool(e["fires"]) for e in g))
         for m, c in cap.items() for t in ("easy", "medium", "hard") for g in [[e for e in c["covert"] if e["tier"] == t]]}
DATA = json.dumps(dict(bars=bars, hist=hist, tiers=tiers))
TEMPLATE = (HERE / "dashboard_template.html").read_text()
Path(sys.argv[1]).write_text(TEMPLATE.replace("__DATA__", DATA))
print("wrote", sys.argv[1])
