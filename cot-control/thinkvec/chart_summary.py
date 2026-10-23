"""Headline chart for a summarized run: strict pass rate, accuracy and median CoT words per condition, per-seed points.

    python chart_summary.py RUN_DIR OUT_PNG [COND ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
SURFACE, TEXT, MUTED, BAR, DOT = "#1a1a19", "#f0efec", "#c3c2b7", "#3987e5", "#d95926"
LABEL = {"org_unsteered": "unsteered", "unsteered": "unsteered"}

run, out = Path(sys.argv[1]), Path(sys.argv[2])
summary = pd.read_csv(run / "summary.csv").set_index("condition")
rec = pd.read_csv(run / "analysis/records.csv")
rec["seed"] = rec.condition.str.extract(r"_s(\d+)$")[0].fillna("0").astype(int)
rec["condition"] = rec.condition.str.replace(r"_s\d+$", "", regex=True)
rec["strict_nometa"] = rec.strict_nometa.fillna(False).astype(bool)
rec["correct"] = rec.correct.fillna(False).astype(bool)
base = next(c for c in summary.index if c.endswith("unsteered"))
keep = set(sys.argv[3:]) or set(summary.index)
conds = [base] + sorted((c for c in summary.index if c != base and c in keep), key=lambda c: ("_m" in c, c))
names = [LABEL.get(c, c.replace("org_", "").replace("think_", "think ").replace("rand_", "random ").replace("_p", " +").replace("_m", " -")) for c in conds]
per_seed = rec.groupby(["condition", "seed"]).agg(strict=("strict_nometa", "mean"), correct=("correct", "mean")).reset_index()
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "axes.grid": True, "grid.color": "#33332f",
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for ax, col, title in zip(axes[:2], ["strict", "correct"], ["Strict compliance rate (meta-discussion excluded)", "Accuracy"]):
    vals = [summary.loc[c, col] / summary.loc[c, "n"] for c in conds]
    ax.bar(range(len(conds)), vals, color=BAR, width=0.6, linewidth=0)
    for i, c in enumerate(conds):
        ps = per_seed[per_seed.condition == c]
        ax.scatter([i] * len(ps), ps[col], color=DOT, s=28, zorder=3)
        ax.text(i, vals[i] + 0.02, f"{int(summary.loc[c, col])}/{int(summary.loc[c, 'n'])}", ha="center", va="bottom", color=TEXT, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, color=TEXT, loc="left")
    ax.set_xticks(range(len(conds)), names, rotation=30, ha="right", fontsize=8)
ax = axes[2]
vals = [summary.loc[c, "median_cot_words"] for c in conds]
ax.bar(range(len(conds)), vals, color=BAR, width=0.6, linewidth=0)
for i, c in enumerate(conds):
    ax.text(i, vals[i], f"{vals[i]:.0f}\n{int(summary.loc[c, 'truncated'])}/{int(summary.loc[c, 'n'])} trunc.", ha="center", va="bottom", color=TEXT, fontsize=8)
ax.set_ylim(0, 1.25 * max(vals))
ax.set_title("Median CoT words (and truncations at 25k)", color=TEXT, loc="left")
ax.set_xticks(range(len(conds)), names, rotation=30, ha="right", fontsize=8)
fig.suptitle(f"Think/no-think vector steering on the Astra cotcontrol eval, run {run.name} "
             f"({'opsd-grpo organism' if base.startswith('org') else 'base Qwen3.8-27B'}); bars = seeds pooled, dots = per seed", color=TEXT, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
