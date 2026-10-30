"""Astra CoT-Control for one model under Self-Instructed CoT elicitation (tinker-runner/selfinstruct.py) vs its User Prompt run.

Primary metric: strict = the CoT obeys the constraint (regex grader, or the LLM judge for ignore_question), meta-reasoning
allowed; under the elicitation methods the graded CoT is the model's continuation only. Per condition: strict passes with a
95% Wilson interval, the strict passes whose CoT has more than --trivial-words words, no-meta passes (secondary), median CoT
words, answer accuracy, truncations, exact McNemar against the first run (paired by item), and the echo count for bootstrap
runs. Then strict passes per constraint. The figure shows both, dark scheme.

    .venv/bin/python astra_elicit.py --title "..." --fig NAME up=DIR upcp=DIR ... > summary.md
"""

import argparse
import csv
import glob
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grading"))
from cotswitch.analyze import wilson  # noqa: E402

matplotlib.use("Agg")
FIGURES = Path(__file__).resolve().parents[3] / "figures"
KEY = ("dataset", "item_idx", "constraint", "sample_idx")
LABELS = {"up": "no trick:\nconstraint in\nprompt only", "upcp": "+ constraint-\nspecific prefill\n(up_cp)",
          "upgp": "+ generic\nprefill\n(up_gp)", "cb": "+ bootstrap\ninstruction, no\nprefill (cb)"}
INK, INK2, MUTED, GRID, AXIS, SURFACE, PAGE, BLUE = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835", "#1a1a19", "#0d0d0d", "#3987e5"


def load(run):
    rec = {tuple(r[k] for k in KEY): r for r in csv.DictReader(open(Path(run) / "analysis/records.csv")) if r["protocol"] == "astra"}
    (gen,) = glob.glob(str(Path(run) / "generations/*/astra.jsonl"))
    for line in open(gen):
        x = json.loads(line)
        r = rec[(x["dataset"], str(x["item_idx"]), x["constraint"], str(x["sample_idx"]))]
        r["words"], r["echo"] = len((x["reasoning"] or "").split()), x.get("echo_stripped")
    return rec


def mcnemar(a, b):
    """a, b: dicts key -> 0/1 on the same keys. Returns (b-only, a-only, exact two-sided p)."""
    bo = sum(1 for k in a if b[k] and not a[k])
    ao = sum(1 for k in a if a[k] and not b[k])
    return bo, ao, (binomtest(bo, bo + ao).pvalue if bo + ao else 1.0)


def figure(data, cons, ok, title, path):
    names = list(data)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1, 1.25]}, facecolor=PAGE)
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    n = [len(data[m]) for m in names]
    k = [sum(map(ok, data[m].values())) for m in names]
    lo, hi = zip(*(wilson(ki, ni) for ki, ni in zip(k, n)))
    rate = [100 * ki / ni for ki, ni in zip(k, n)]
    x = np.arange(len(names))
    a1.bar(x, rate, width=0.55, color=BLUE, zorder=3)
    a1.vlines(x, [100 * v for v in lo], [100 * v for v in hi], color=INK2, lw=1.5, zorder=4)
    for xi, r, h, ki, ni in zip(x, rate, hi, k, n):
        a1.text(xi, 100 * h + 1.2, f"{ki}/{ni}", ha="center", va="bottom", color=INK, fontsize=10)
    a1.set_xticks(x, [LABELS.get(m, m) for m in names], fontsize=9)
    a1.set_ylabel("strict compliance, % of items (95% Wilson)", color=INK2, fontsize=10)
    a1.set_ylim(0, max(30, 100 * max(hi) + 8))
    a1.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in a1.spines.values():
        s.set_visible(False)
    a1.spines["bottom"].set_visible(True)
    a1.spines["bottom"].set_color(AXIS)
    a1.set_title("Pooled over 9 constraints x 12 items", color=INK, fontsize=11, loc="left")
    grid = np.array([[sum(ok(r) for kk, r in data[m].items() if kk[2] == c) for m in names] for c in cons])
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, "#184f95", BLUE, "#b7d3f6"])
    a2.imshow(grid, cmap=cmap, vmin=0, vmax=12, aspect="auto")
    for i in range(len(cons)):
        for j in range(len(names)):
            v = grid[i, j]
            a2.text(j, i, f"{v}", ha="center", va="center", fontsize=10, color=PAGE if v >= 8 else INK)
    a2.set_xticks(range(len(names)), [LABELS.get(m, m) for m in names], fontsize=8.5)
    a2.set_yticks(range(len(cons)), [c.replace("_", " ") for c in cons], fontsize=9)
    a2.set_xticks(np.arange(len(names)) - 0.5, minor=True)
    a2.set_yticks(np.arange(len(cons)) - 0.5, minor=True)
    a2.grid(which="minor", color=PAGE, lw=2)
    a2.tick_params(which="minor", length=0)
    for s in a2.spines.values():
        s.set_visible(False)
    a2.set_title("Strict passes per constraint (of 12)", color=INK, fontsize=11, loc="left")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=150, facecolor=PAGE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=run_dir; the first is the User Prompt run the others are paired against")
    ap.add_argument("--trivial-words", type=int, default=5)
    ap.add_argument("--title", required=True)
    ap.add_argument("--fig", required=True, help="figure file name under mats12_sprint/figures/")
    args = ap.parse_args()
    data = {name: load(run) for name, run in (s.split("=", 1) for s in args.runs)}
    base = next(iter(data.values()))
    ok = lambda r: r["strict"] == "1"
    nm = lambda r: r["strict_nometa"] == "1"
    real = lambda r: ok(r) and r["words"] > args.trivial_words
    print(f"| condition | strict passes (95% Wilson) | passes, CoT > {args.trivial_words} words | no-meta passes | median CoT words | "
          f"accuracy | truncated | echoed | vs first: this-only / first-only, p |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, d in data.items():
        assert set(d) == set(base), name
        k, n = sum(map(ok, d.values())), len(d)
        lo, hi = wilson(k, n)
        echo = sum(r["echo"] is True for r in d.values())
        row = (f"| {name} | {k}/{n} ({100 * lo:.1f}-{100 * hi:.1f}%) | {sum(map(real, d.values()))} | {sum(map(nm, d.values()))} | "
               f"{st.median(r['words'] for r in d.values()):.0f} | {sum(r['correct'] in ('True', '1', '1.0') for r in d.values())}/{n} | "
               f"{sum(r['finish_reason'] == 'length' for r in d.values())} | {echo if name.startswith('cb') else '-'} |")
        if d is base:
            print(row + " - |")
            continue
        mo, bo, p = mcnemar({kk: ok(base[kk]) for kk in base}, {kk: ok(d[kk]) for kk in base})
        print(row + f" {mo} / {bo}, p {p:.3g} |")
    cons = sorted({kk[2] for kk in base})
    print("\n| constraint | " + " | ".join(data) + " |")
    print("|---|" + "---|" * len(data))
    for c in cons:
        print(f"| {c} | " + " | ".join(str(sum(ok(r) for kk, r in d.items() if kk[2] == c)) for d in data.values()) + " |")
    path = FIGURES / args.fig
    figure(data, cons, ok, args.title, path)
    print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
