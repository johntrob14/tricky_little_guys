"""Per cell, from logs/<cell>/monitor_<part>_cot_only/<cell>/summary.json (part visible by default, or --part reasoning) and the cell's .eval scores: episodes, episodes with at least one scored turn
(visible text present), mean per-episode max suspicion over those (with SE), the same split by leaked / not leaked, and the mean over scored turns.
Writes results.json (one row per episode) and prints a table. Counts and means only; no text.

    .venv/bin/python results/monitor_visible/summarize.py [--part reasoning] [--exclude-final] CELL [CELL ...]

--exclude-final drops each episode's last assistant turn (usually the submit turn and its summary) before taking the max. Output results.json, or
results_nofinal.json, results_reasoning.json, results_reasoning_nofinal.json.
"""

import json
import statistics
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[2]


def mean_se(xs):
    if not xs:
        return "-"
    se = statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0
    return f"{statistics.fmean(xs):.2f} +- {se:.2f} ({len(xs)})"


args = sys.argv[1:]
exclude_final = "--exclude-final" in args
part = args[args.index("--part") + 1] if "--part" in args else "visible"
cells = [a for i, a in enumerate(args) if a not in ("--exclude-final", "--part") and not (i and args[i - 1] == "--part")]
mdir = f"monitor_{part}_cot_only"
rows = []
print("| cell | episodes | episodes with a scored turn | mean episode max (n) | leaked: mean episode max (n) | not leaked: mean episode max (n) | mean over scored turns (n) | unparsed |")
print("|---|---|---|---|---|---|---|---|")
for cell in cells:
    d = ROOT / "logs" / cell
    summ = json.loads((d / mdir / cell / "summary.json").read_text())
    turns = [json.loads(l) for l in (d / mdir / cell / "monitor.jsonl").read_text().splitlines()]
    leaked = {}
    for p in d.glob("*.eval"):
        for s in read_eval_log(str(p)).samples or []:
            leaked[(str(s.id), s.epoch)] = bool(next(iter(s.scores.values())).metadata.get("side_leaked"))
    by_ep = {}
    for t in turns:
        by_ep.setdefault((t["sample"], t["epoch"], t["log"]), []).append(t)
    eps = []
    for e in summ["episodes"]:
        ts = by_ep[(e["sample"], e["epoch"], e["log"])]
        last = max(t["turn"] for t in ts)
        sc = [t["score"] for t in ts if t["score"] is not None and not (exclude_final and t["turn"] == last)]
        assert exclude_final or (max(sc) if sc else None) == e["max"]
        eps.append(dict(cell=cell, sample=e["sample"], epoch=e["epoch"], leaked=leaked[(e["sample"], e["epoch"])], max=max(sc) if sc else None, scored_turns=len(sc), turns=e["turns"]))
    assert len(eps) == len(leaked), f"{cell}: {len(eps)} monitored episodes, {len(leaked)} in the logs"
    rows += eps
    mx = [e["max"] for e in eps if e["max"] is not None]
    scored = [t["score"] for ts in by_ep.values() for t in ts if t["score"] is not None and not (exclude_final and t["turn"] == max(x["turn"] for x in ts))]
    print(f"| {cell} | {len(eps)} | {len(mx)}/{len(eps)} | {mean_se(mx)} | {mean_se([e['max'] for e in eps if e['leaked'] and e['max'] is not None])} | "
          f"{mean_se([e['max'] for e in eps if not e['leaked'] and e['max'] is not None])} | {mean_se(scored)} | {summ['unparsed']} |")
(Path(__file__).parent / ("results" + ("_reasoning" if part == "reasoning" else "") + ("_nofinal" if exclude_final else "") + ".json")).write_text(json.dumps(rows, indent=1))
