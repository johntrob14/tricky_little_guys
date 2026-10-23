"""Headline chart for a summarized run: strict compliance rate, accuracy and median CoT words against steering strength.

    python chart_summary.py RUN_DIR OUT_PNG

The x axis is the signed per-layer alpha of the main layer set (the one with most conditions; 0 = unsteered). The think
direction is a line with filled circles (seeds pooled) and small per-seed dots; the matched-norm random direction is
squares at its alphas; other layer sets (e.g. block 35 alone) sit at the two ends as diamonds. Condition names follow
steer.py: [org_](think|rand)_<layers>_(p|m)<alpha>[_s<seed>].
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
SURFACE, TEXT, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#33332f"
C_THINK, C_RAND, C_OTHER, C_REF = "#d95926", "#199e70", "#d55181", "#3987e5"  # dark-mode categorical slots 2, 3, 5, 1
BASE_ACC = 64 / 108  # base Qwen3.8-27B, unsteered, results/thinkvec/main

run, out = Path(sys.argv[1]), Path(sys.argv[2])
summary = pd.read_csv(run / "summary.csv").set_index("condition")
rec = pd.read_csv(run / "analysis/records.csv")
rec["seed"] = rec.condition.str.extract(r"_s(\d+)$")[0].fillna("0").astype(int)
rec["condition"] = rec.condition.str.replace(r"_s\d+$", "", regex=True)
rec["strict"] = rec.strict.fillna(False).astype(bool)
rec["correct"] = rec.correct.fillna(False).astype(bool)
per_seed = rec.groupby(["condition", "seed"]).agg(strict=("strict", "mean"), correct=("correct", "mean")).reset_index()


def parse(c):
    m = re.fullmatch(r"(?:org_)?([a-z_]+?)_(L[\d-]+(?:s\d+)?)_(p|m)([\d.]+)", c)
    return None if m is None else (m[1], m[2], (1 if m[3] == "p" else -1) * float(m[4]))


base = next(c for c in summary.index if c.endswith("unsteered"))
parsed = {c: parse(c) for c in summary.index if c != base}
tags = Counter(p[1] for p in parsed.values())
main_tag = max(tags, key=lambda t: (tags[t], len(t)))
alphas = sorted({p[2] for p in parsed.values() if p[1] == main_tag} | {0.0})
others = sorted({(p[1], p[2]) for p in parsed.values() if p[1] != main_tag})
cats = [o for o in others if o[1] < 0] + alphas + [o for o in others if o[1] > 0]
xpos = {cat: i for i, cat in enumerate(cats)}
ticks = [f"{t.replace('L', 'block ')}\nalpha {a:+g}" if isinstance(cat, tuple) else ("0\n(unsteered)" if a == 0 else f"{a:+g}")
         for cat in cats for (t, a) in [cat if isinstance(cat, tuple) else (None, cat)]]
layers_text = main_tag.replace("L", "blocks ").replace("-", " to ").replace("s4", " step 4")


def x_of(c):
    if c == base:
        return xpos[0.0]
    fam, tag, a = parsed[c]
    return xpos[a] if tag == main_tag else xpos[(tag, a)]


def value(c, col):
    return summary.loc[c, col] / summary.loc[c, "n"] if col in ("strict", "correct") else summary.loc[c, col]


plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": TEXT, "ytick.color": MUTED, "text.color": TEXT, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
panels = [("strict", "Strict compliance rate (meta-reasoning allowed)", (0, 1)), ("correct", "Accuracy", (0, 1)),
          ("median_cot_words", "Median CoT words", None)]
rand_conds = [c for c, p in parsed.items() if p[0].startswith("rand") and p[1] == main_tag]
vec_name = next(p[0] for p in parsed.values() if not p[0].startswith("rand"))
think_conds = [base] + [c for c, p in parsed.items() if p[0] == vec_name and p[1] == main_tag]
other_conds = [c for c, p in parsed.items() if p[1] != main_tag]
for ax, (col, title, ylim) in zip(axes, panels):
    ax.axhline(value(base, col), color=C_REF, linestyle=":", linewidth=1.2, zorder=1)
    pts = sorted(((x_of(c), value(c, col), c) for c in think_conds), key=lambda t: t[0])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C_THINK, linewidth=2, zorder=2)
    for x, y, c in pts:
        ax.scatter([x], [y], color=C_THINK if c != base else C_REF, s=90, zorder=4, edgecolors=SURFACE, linewidths=1.5)
    for c in rand_conds:
        ax.scatter([x_of(c) + 0.18], [value(c, col)], color=C_RAND, marker="s", s=80, zorder=4, edgecolors=SURFACE, linewidths=1.5)
    for c in other_conds:
        ax.scatter([x_of(c)], [value(c, col)], color=C_OTHER, marker="D", s=80, zorder=4, edgecolors=SURFACE, linewidths=1.5)
    if col in ("strict", "correct"):
        for c in summary.index:
            ps = per_seed[per_seed.condition == c]
            if len(ps) > 1:
                color = C_RAND if c in rand_conds else C_OTHER if c in other_conds else C_THINK if c != base else C_REF
                ax.scatter([x_of(c) + (0.18 if c in rand_conds else 0)] * len(ps), ps[col], color=color, s=34, alpha=0.75, zorder=3, linewidths=0)
        for c in summary.index:
            dx = 0.18 if c in rand_conds else 0
            ax.annotate(f"{int(summary.loc[c, col])}/{int(summary.loc[c, 'n'])}", (x_of(c) + dx, value(c, col)), textcoords="offset points",
                        xytext=(0, 9 if c not in rand_conds else -14), ha="center", fontsize=7.5, color=TEXT)
        ax.set_ylim(*ylim)
    else:
        for c in summary.index:
            dx = 0.18 if c in rand_conds else 0
            tr, n = int(summary.loc[c, "truncated"]), int(summary.loc[c, "n"])
            ax.annotate(f"{value(c, col):.0f}" + (f"\n{tr}/{n} at cap" if tr / n >= 0.02 else ""), (x_of(c) + dx, value(c, col)),
                        textcoords="offset points", xytext=(0, 9 if c not in rand_conds else -16), ha="center", fontsize=7.5, color=TEXT)
        ax.set_ylim(0, 1.3 * max(value(c, col) for c in summary.index))
    if col == "correct":
        ax.axhline(BASE_ACC, color=MUTED, linestyle="--", linewidth=1, zorder=1)
        ax.annotate("base model, unsteered", (-0.5, BASE_ACC), textcoords="offset points", xytext=(2, 4), ha="left", fontsize=7.5, color=MUTED)
    ax.set_title(title, color=TEXT, loc="left")
    ax.set_xticks(range(len(cats)), ticks, fontsize=8.5)
    ax.set_xlim(-0.6, len(cats) - 0.4)
fig.supxlabel(f"signed per-layer alpha in units of the {vec_name} vector's own norm, {layers_text}" + ("; ends: other layer set" if other_conds else ""), color=MUTED, fontsize=9)
handles = [plt.Line2D([], [], color=C_THINK, marker="o", markersize=8, linewidth=2, label=f"{vec_name} direction, {layers_text}"),
           plt.Line2D([], [], color=C_RAND, marker="s", markersize=8, linewidth=0, label="random direction, same per-layer norms"),
           plt.Line2D([], [], color=C_OTHER, marker="D", markersize=8, linewidth=0, label=f"{vec_name} direction, other layer set"),
           plt.Line2D([], [], color=C_REF, marker="o", markersize=8, linewidth=0, label="unsteered (dotted line: its level)"),
           plt.Line2D([], [], color=TEXT, marker="o", markersize=5, linewidth=0, alpha=0.75, label="small dots: per-seed values")]
if not other_conds:
    handles.pop(2)
fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, labelcolor=TEXT, fontsize=9, bbox_to_anchor=(0.5, 0.93))
model = "opsd-grpo organism (Qwen3.8-27B + opsd-grpo1b LoRA)" if base.startswith("org") else "base Qwen3.8-27B"
fig.suptitle(f"{vec_name} vector steering during the Astra cotcontrol eval: {model}, run {run.name}", color=TEXT, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.02, 1, 0.88))
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
