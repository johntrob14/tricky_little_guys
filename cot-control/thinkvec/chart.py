"""Dark-theme chart of a summarized run: strict passes and partial compliance per constraint by condition, and CoT
length by condition.

    python chart.py RUN_DIR OUT_PNG
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
SURFACE, TEXT, MUTED = "#1a1a19", "#f0efec", "#c3c2b7"
COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]  # categorical slots, dark steps, fixed order
ORDER = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking",
         "alternating_case", "uppercase_thinking", "end_of_sentence", "ignore_question"]

run, out = Path(sys.argv[1]), Path(sys.argv[2])
summary = pd.read_csv(run / "summary.csv").set_index("condition")
per_c = pd.read_csv(run / "summary_constraints.csv")
conds = ["unsteered"] + sorted((c for c in summary.index if c != "unsteered"), key=lambda c: ("_m" in c, c))  # unsteered, +alpha, -alpha
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "axes.grid": True, "grid.color": "#33332f",
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [3, 3, 1.3]})
w = 0.8 / len(conds)
for ax, col, title in zip(axes[:2], ["strict", "partial"], ["Strict passes (of 12 items per constraint)", "Partial compliance (mean, 0-1)"]):
    for j, c in enumerate(conds):
        d = per_c[per_c.condition == c].set_index("constraint").reindex(ORDER)
        label = f"{c}: {int(summary.loc[c, 'strict'])}/{int(summary.loc[c, 'n'])} strict" if col == "strict" else c
        ax.bar([i + (j - (len(conds) - 1) / 2) * w for i in range(len(ORDER))], d[col].fillna(0), width=w * 0.92, color=COLORS[j], label=label, linewidth=0)
    ax.set_xticks(range(len(ORDER)), ORDER, rotation=35, ha="right")
    ax.set_ylim(0, 12 if col == "strict" else 1.05)
    ax.set_title(title, color=TEXT, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT)
ax = axes[2]
ax.bar(range(len(conds)), [summary.loc[c, "median_cot_words"] for c in conds], color=COLORS[: len(conds)], width=0.7, linewidth=0)
for i, c in enumerate(conds):
    ax.text(i, summary.loc[c, "median_cot_words"], f"{summary.loc[c, 'median_cot_words']:.0f}\n{int(summary.loc[c, 'truncated'])} trunc.", ha="center", va="bottom", color=TEXT, fontsize=9)
ax.set_xticks(range(len(conds)), conds, rotation=20, ha="right")
ax.set_ylim(0, 1.2 * max(summary.loc[c, "median_cot_words"] for c in conds))
ax.set_title("Median CoT words", color=TEXT, loc="left")
fig.suptitle(f"Qwen3.8-27B, Astra cotcontrol eval (108 items, effort xhigh, T 1.0): think/no-think vector steering, run {run.name}", color=TEXT, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
