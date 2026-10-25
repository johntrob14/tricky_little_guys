"""Mann-Whitney on per-episode max and mean (episodes doing the side task: attempted or leaked) and Fisher on attempt and leak counts, organism vs base,
from visible_luna.json and reasoning_luna.json. Writes stats.txt. Run with a python that has scipy (the arena venv has none): python stats.py"""
import json
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

HERE = Path(__file__).parent
rows = {}
for part in ("visible", "reasoning"):
    for r in json.loads((HERE / f"{part}_luna.json").read_text()):
        rows.setdefault((r["cell"], part), []).append(r)
ORG_OFF, ORG_ON, BASE_OFF, BASE_ON = "ds41rw-nocot-cvuser-easy", "ds41rw-think25-cvuser-easy", "ds41or-nothink-cvuser-easy", "ds41or-cvuser-easy"
out = []
for a, b, part in [(ORG_OFF, BASE_OFF, "visible"), (ORG_ON, BASE_ON, "visible"), (ORG_ON, BASE_ON, "reasoning")]:
    for k in ("max", "mean"):
        x, y = ([r[k] for r in rows[(c, part)] if (r["attempted"] or r["leaked"]) and r[k] is not None] for c in (a, b))
        p = mannwhitneyu(x, y).pvalue if len(set(x + y)) > 1 else float("nan")
        out.append(f"{part:9s} {k:4s} {a} {sum(x) / len(x):.2f} (n={len(x)}) vs {b} {sum(y) / len(y):.2f} (n={len(y)}): Mann-Whitney p={p:.3g}")
for a, b in [(ORG_OFF, BASE_OFF), (ORG_ON, BASE_ON)]:
    for k in ("attempted", "leaked"):
        x, y = (sum(r[k] for r in rows[(c, "visible")]) for c in (a, b))
        n, m = len(rows[(a, "visible")]), len(rows[(b, "visible")])
        out.append(f"{k:9s} {a} {x}/{n} vs {b} {y}/{m}: Fisher p={fisher_exact([[x, n - x], [y, m - y]]).pvalue:.3g}")
(HERE / "stats.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
