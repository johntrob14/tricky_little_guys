"""Dark-theme chart of a summarized run: strict passes and partial compliance per constraint by condition, and CoT
length by condition.

    python chart.py RUN_DIR OUT_PNG [COND ...]     # optional: only these conditions (the unsteered one is always included), at most 5
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
base = next(c for c in summary.index if c.endswith("unsteered"))
keep = set(sys.argv[3:]) or set(summary.index)
conds = [base] + sorted((c for c in summary.index if c != base and c in keep), key=lambda c: ("_m" in c, c))  # unsteered, +alpha, -alpha
assert len(conds) <= len(COLORS), f"{len(conds)} conditions, {len(COLORS)} colours: pass a subset"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "axes.grid": True, "grid.color": "#33332f",
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), gridspec_kw={"width_ratios": [3, 3, 1.6]})
w = 0.8 / len(conds)
seeds = summary.seeds if "seeds" in summary else pd.Series(1, index=summary.index)
for ax, col, title in zip(axes[:2], ["strict_rate", "partial"], ["Length-adjusted strict pass rate per constraint, meta-reasoning allowed (12 items x seeds)", "Partial compliance (mean, 0-1)"]):
    for j, c in enumerate(conds):
        d = per_c[per_c.condition == c].set_index("constraint").reindex(ORDER)
        d["strict_rate"] = d["strict_adj"] / (12 * seeds[c])
        label = f"{c}: {int(summary.loc[c, 'strict_adj'])}/{int(summary.loc[c, 'n'])} strict (adj.)" if col == "strict_rate" else c
        ax.bar([i + (j - (len(conds) - 1) / 2) * w for i in range(len(ORDER))], d[col].fillna(0), width=w * 0.92, color=COLORS[j], label=label, linewidth=0)
    ax.set_xticks(range(len(ORDER)), ORDER, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, color=TEXT, loc="left")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=min(len(conds), 3), frameon=False, labelcolor=TEXT, bbox_to_anchor=(0.5, 0.93), fontsize=9)
ax = axes[2]
ax.bar(range(len(conds)), [summary.loc[c, "median_cot_words"] for c in conds], color=COLORS[: len(conds)], width=0.7, linewidth=0)
for i, c in enumerate(conds):
    ax.text(i, summary.loc[c, "median_cot_words"], f"{summary.loc[c, 'median_cot_words']:.0f}\n{int(summary.loc[c, 'truncated'])} trunc.", ha="center", va="bottom", color=TEXT, fontsize=9)
ax.set_xticks(range(len(conds)), conds, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 1.2 * max(summary.loc[c, "median_cot_words"] for c in conds))
ax.set_title("Median CoT words", color=TEXT, loc="left")
fig.suptitle(f"Astra cotcontrol eval (108 items x seeds, effort xhigh, T 1.0): think/no-think vector steering, run {run.name} "
             f"({'opsd-grpo organism' if conds[0].startswith('org') else 'base Qwen3.8-27B'})", color=TEXT, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
