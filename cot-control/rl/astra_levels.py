"""Astra CoT-Control (108 items) for base and the length-trained models at each reasoning level.

Per (model, level): passes without meta-discussion (strict_nometa), passes with meta allowed (strict), the no-meta passes whose
CoT has more than --trivial-words words (a CoT of a few words passes case, end-of-sentence and suppression rules by containing
no reasoning), median CoT words, answer accuracy, truncations, and an exact McNemar test against base at the same level
(paired by item). Then no-meta passes per constraint.

    .venv/bin/python astra_levels.py > ../results/astra_levels/summary.md
"""

import argparse
import csv
import json
import statistics as st
from pathlib import Path

from scipy.stats import binomtest

RUNS = Path(__file__).parent.parent / "results/tinker-runs"
LEVELS = ["low", "medium", "high"]
MODELS = {  # model -> level -> run dir; medium runs used the template default effort and max_new_tokens 25,000
    "base": {"low": "astra-base-low-t1", "medium": "gptoss120b-t1", "high": "astra-base-high-t1"},
    "len2": {"low": "astra-len2-low-t1", "medium": "len2-step60-t1", "high": "astra-len2-high-t1"},
    "eff1": {"low": "astra-eff1-low-t1", "medium": "eff1-step40-t1", "high": "astra-eff1-high-t1"},
    "eff2": {"low": "astra-eff2-low-t1", "medium": "eff2-step40-t1", "high": "astra-eff2-high-t1"},
    "effm1": {"low": "astra-effm1-low-t1", "medium": "effm1-step30-t1", "high": "astra-effm1-high-t1"},
    "effm2": {"low": "astra-effm2-low-t1", "medium": "effm2-step50-t1", "high": "astra-effm2-high-t1"},
}
KEY = ("dataset", "item_idx", "constraint", "sample_idx")


def load(run):
    rec = {tuple(r[k] for k in KEY): r for r in csv.DictReader(open(RUNS / run / "analysis/records.csv")) if r["protocol"] == "astra"}
    words = {}
    for line in open(RUNS / run / "generations/gpt-oss-120b/astra.jsonl"):
        x = json.loads(line)
        words[(x["dataset"], str(x["item_idx"]), x["constraint"], str(x["sample_idx"]))] = len((x["reasoning"] or "").split())
    for k, r in rec.items():
        r["words"] = words[k]
    return rec


def mcnemar(a, b):
    """a, b: dicts key -> 0/1 on the same keys. Returns (b-only, a-only, exact two-sided p)."""
    bo = sum(1 for k in a if b[k] and not a[k])
    ao = sum(1 for k in a if a[k] and not b[k])
    return bo, ao, (binomtest(bo, bo + ao).pvalue if bo + ao else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trivial-words", type=int, default=5)
    args = ap.parse_args()
    data = {m: {e: load(run) for e, run in lv.items() if (RUNS / run / "analysis/records.csv").exists()} for m, lv in MODELS.items()}
    nm = lambda r: r["strict_nometa"] == "1"
    real = lambda r: nm(r) and r["words"] > args.trivial_words
    for e in LEVELS:
        print(f"## {e}\n")
        print(f"| model | no-meta passes | meta allowed | no-meta, CoT > {args.trivial_words} words | median CoT words | accuracy | truncated | "
              f"vs base: model-only / base-only, p | same, CoT > {args.trivial_words} words |")
        print("|---|---|---|---|---|---|---|---|---|")
        b = data["base"][e]
        for m in MODELS:
            if e not in data[m]:
                print(f"| {m} | not run |"); continue
            d = data[m][e]
            assert set(d) == set(b), (m, e)
            row = f"| {m} | {sum(map(nm, d.values()))} | {sum(r['strict'] == '1' for r in d.values())} | {sum(map(real, d.values()))} | " \
                  f"{st.median(r['words'] for r in d.values()):.0f} | {sum(r['correct'] in ('True', '1', '1.0') for r in d.values())} | " \
                  f"{sum(r['finish_reason'] == 'length' for r in d.values())} |"
            if m == "base":
                print(row + " - | - |"); continue
            mo, bo, p = mcnemar({k: nm(b[k]) for k in b}, {k: nm(d[k]) for k in b})
            mo2, bo2, p2 = mcnemar({k: real(b[k]) for k in b}, {k: real(d[k]) for k in b})
            print(row + f" {mo} / {bo}, p {p:.3f} | {mo2} / {bo2}, p {p2:.3f} |")
        print()
    cons = sorted({k[2] for k in data["base"]["low"]})
    print("## No-meta passes per constraint (of 12), low / medium / high\n")
    print("| constraint | " + " | ".join(MODELS) + " |")
    print("|---|" + "---|" * len(MODELS))
    for c in cons:
        cells = []
        for m in MODELS:
            cells.append(" / ".join(str(sum(nm(r) for k, r in data[m][e].items() if k[2] == c)) if e in data[m] else "-" for e in LEVELS))
        print(f"| {c} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
