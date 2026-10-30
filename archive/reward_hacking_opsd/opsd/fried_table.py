"""Tabulate the fried-model-organisms outputs across models.
python3 opsd/fried_table.py --root results/fried [--models base A B C] > results/fried.md
Reads <root>/evalsuite/<name>/summary.json (benchmarks.* numeric leaves, flattened to dotted keys) and
<root>/decisiveness/<name>/panel.json (<metric>.point). Missing files or errors are shown as '-'."""
import argparse, json, os

PANEL = ["decisiveness", "order_consistency", "transitivity_fas", "transitivity_triad", "q_agreement", "unidim_fit_brier"]


def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/fried")
    ap.add_argument("--models", nargs="+", default=["base", "A", "B", "C"])
    ap.add_argument("--decis-suffix", default="_nothink", help="suffix of the decisiveness run dir to prefer")
    a = ap.parse_args()
    rows = {}
    for m in a.models:
        r = {}
        # evalsuite writes <name>/summary.json (ifeval, perplexity, safety) and <name>_mmlu/summary.json (limited MMLU)
        # A partial rerun (e.g. --benchmarks safety) overwrites summary.json with only that block, after
        # fried_chain.sh backs the old file up as summary_prev.json; so read prev first and let the newer file win.
        for sub in (m, m + "_mmlu"):
            for fn in ("summary_prev.json", "summary.json"):
                s = load(os.path.join(a.root, "evalsuite", sub, fn))
                r.update(flat((s or {}).get("benchmarks", {}), ""))
                for k, v in ((s or {}).get("benchmarks") or {}).items():
                    r.pop(f"{k}.error", None)
                    if isinstance(v, dict) and "error" in v:
                        r[f"{k}.error"] = v["error"]
        # preference panels: prefer the thinking-off run (<name>_nothink), fall back to the plain name
        p = load(os.path.join(a.root, "decisiveness", m + a.decis_suffix, "panel.json")) or \
            load(os.path.join(a.root, "decisiveness", m, "panel.json"))
        for k in PANEL:
            if p and isinstance(p.get(k), dict) and "point" in p[k]:
                r[f"panel.{k}"] = p[k]["point"]
        rows[m] = r
    keys = sorted({k for r in rows.values() for k in r if not k.endswith(".error")})
    print("| metric | " + " | ".join(a.models) + " |")
    print("|---|" + "---|" * len(a.models))
    for k in keys:
        cells = []
        for m in a.models:
            v = rows[m].get(k)
            cells.append("-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)))
        print(f"| {k} | " + " | ".join(cells) + " |")
    errs = [(m, k, v) for m, r in rows.items() for k, v in r.items() if k.endswith(".error")]
    if errs:
        print("\nErrors:")
        for m, k, v in errs:
            print(f"- {m} {k}: {v}")


if __name__ == "__main__":
    main()
