"""Both steering vectors on the organism, with steering on all generated tokens vs on the CoT only: length-adjusted strict
compliance (meta-reasoning allowed), accuracy and median CoT words at -alpha, 0, +alpha, plus the matched-norm random direction.

    python compare_chart.py OUT_PNG

Runs: org4 (think, all generated tokens), org5 (cotout, all generated tokens; the intended CoT-only mask did not engage, see
README), org6_cotout (cotout, CoT only; one condition, +0.2 seed 0, before the rerun was stopped). All share the organism's
unsteered rows.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from lenadj import add_length_adjustment, adjusted_rate

matplotlib.use("Agg")
R = Path(__file__).resolve().parent.parent / "results/thinkvec"
RUNS = [("org4", "think", 0.125, "all generated tokens"), ("org5", "cotout", 0.2, "all generated tokens"), ("org6_cotout", "cotout", 0.2, "CoT only")]
SURFACE, TEXT, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#33332f"
C_ALL, C_COT, C_RAND, C_REF = "#d95926", "#3987e5", "#199e70", "#c3c2b7"


def load(run):
    rec = pd.read_csv(R / run / "analysis/records.csv")
    gens = {}
    for f in (R / run / "generations").glob("*/astra.jsonl"):
        for line in open(f):
            r = json.loads(line)
            gens[(r["condition"], r["key"])] = r
    rec["cot_words"] = [len((gens[(c, k)]["reasoning"] or "").split()) for c, k in zip(rec.condition, rec.key)]
    rec["seed"] = rec.condition.str.extract(r"_s(\d+)$")[0].fillna("0").astype(int)
    rec["condition"] = rec.condition.str.replace(r"_s\d+$", "", regex=True)
    rec["strict"] = rec.strict.fillna(False).astype(bool)
    rec["correct"] = rec.correct.fillna(False).astype(bool)
    add_length_adjustment(rec, "org_unsteered")
    return rec


rows = []
for run, vec, alpha, mode in RUNS:
    rec = load(run)
    uns = rec[rec.condition == "org_unsteered"]
    for cond, g in rec.groupby("condition"):
        m = re.fullmatch(r"org_([a-z_]+?)_L12-60s4_(p|m)([\d.]+)", cond)
        if cond == "org_unsteered":
            fam, a = "unsteered", 0.0
        elif m and float(m[3]) == alpha:
            fam, a = ("random" if m[1].startswith("rand") else vec), (1 if m[2] == "p" else -1) * float(m[3])
        else:
            continue
        seeds = [adjusted_rate(gs, uns[uns.seed == s]) for s, gs in g.groupby("seed")]
        rows.append(dict(run=run, vec=vec, mode=mode, fam=fam, alpha=a, n=len(g), strict_adj=adjusted_rate(g, uns), strict_raw=g.strict.mean(),
                         seeds=seeds, correct=g.correct.mean(), cot_words=g.cot_words.median(), truncated=int((g.finish_reason == "length").sum())))
df = pd.DataFrame(rows)
df.drop(columns=["seeds"]).to_csv(R / "compare_vectors_masks.csv", index=False)
print(df.drop(columns=["seeds"]).round(3).to_string())

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": TEXT,
                     "ytick.color": MUTED, "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(1, 4, figsize=(21, 5.8), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
for vi, (vec, alpha) in enumerate((("think", 0.125), ("cotout", 0.2))):
    for col, ax, title, ylim in (("strict_adj", axes[2 * vi], "length-adjusted strict compliance", (0.2, 0.62)),
                                 ("correct", axes[2 * vi + 1], "accuracy", (0.35, 0.65))):
        d = df[df.vec == vec]
        base = d[d.fam == "unsteered"].iloc[0]
        ax.axhline(base[col], color=C_REF, linestyle=":", linewidth=1.2)
        for mode, color, dx in (("all generated tokens", C_ALL, -0.06), ("CoT only", C_COT, 0.06)):
            dm = d[(d["mode"] == mode) & d.fam.isin([vec, "unsteered"])].sort_values("alpha")
            xs = [{-alpha: -1, 0.0: 0, alpha: 1}[a] + dx for a in dm.alpha]
            ax.plot(xs, dm[col], color=color, linewidth=2, marker="o", markersize=8, label=f"{vec} vector, steered on {mode}", zorder=3)
            for x, (_, r) in zip(xs, dm.iterrows()):
                if col == "strict_adj" and len(r.seeds) > 1:
                    ax.scatter([x] * len(r.seeds), r.seeds, color=color, s=22, alpha=0.6, zorder=2, linewidths=0)
                ax.annotate(f"{r[col]:.2f}", (x, r[col]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.5, color=TEXT)
            rr = d[(d["mode"] == mode) & (d.fam == "random")]
            for _, r in rr.iterrows():
                x = (1 if r.alpha > 0 else -1) + dx * 3
                ax.scatter([x], [r[col]], marker="s", s=70, color=C_RAND, edgecolors=color, linewidths=1.5, zorder=4)
        ax.set_xticks([-1, 0, 1], [f"-{alpha:g}", "0\n(unsteered)", f"+{alpha:g}"])
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(*ylim)
        ax.set_title(f"{vec} vector: {title}", loc="left", fontsize=10)
handles = [plt.Line2D([], [], color=C_ALL, marker="o", markersize=8, linewidth=2, label="steered on all generated tokens (CoT + answer)"),
           plt.Line2D([], [], color=C_COT, marker="o", markersize=8, linewidth=2, label="steered on CoT tokens only"),
           plt.Line2D([], [], color=C_RAND, marker="s", markersize=8, linewidth=0, label="random direction, same per-layer norms (edge colour = mask)"),
           plt.Line2D([], [], color=C_REF, linestyle=":", label="unsteered"),
           plt.Line2D([], [], color=TEXT, marker="o", markersize=5, linewidth=0, alpha=0.6, label="small dots: per seed")]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, labelcolor=TEXT, fontsize=9, bbox_to_anchor=(0.5, 0.93))
fig.suptitle("opsd-grpo organism, Astra cotcontrol eval (108 items x 3 seeds; CoT-only point: 1 seed), blocks 12..60: think vector (alpha 0.125) vs sentence-copy vector (alpha 0.2)",
             color=TEXT, x=0.01, ha="left", fontsize=11)
fig.supxlabel("signed per-layer alpha in units of each vector's own norm (summed injected norm: sentence-copy 0.93x think); strict compliance: meta-reasoning allowed, length-adjusted by a logistic fit on the unsteered rows",
              color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 0.86))
out = Path(sys.argv[1])
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
