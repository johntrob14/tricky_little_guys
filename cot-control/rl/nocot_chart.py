"""No-CoT accuracy per model (item_difficulty.py --nocot-only: 108 transfer items x 24 samples, empty analysis channel, 'ANSWER:' prefilled, T 1).

    python nocot_chart.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS.parent.parent.parent / "figures/nocot_accuracy.png"
SURFACE, TEXT, TEXT2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
MODELS = [("base", "base", "#3987e5"), ("grpo3 s100", "grpo3-s100", "#d95926"), ("manytypes1b s150", "mt1b-s150", "#199e70"), ("manytypes2 s100", "manytypes2-s100", "#c98500"),
          ("manytypes2c s140", "manytypes2c-s140", "#d55181"), ("manytypes2d s140", "manytypes2d-s140", "#008300"), ("manytypes2e s170", "manytypes2e-s170", "#9085e9")]
GROUPS = [("all 108", None), ("GPQA (4 options)", "gpqa"), ("HLE", "hle"), ("MMLU-Pro (10 options)", "mmlu_pro")]


def load(f):
    return {(r["dataset"], r["item_idx"]): r["nocot_correct"] / r["samples"] for r in map(json.loads, open(RESULTS / "difficulty" / f"nocot_{f}.jsonl"))}


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=11, color=TEXT, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT2)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=TEXT2, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_visible(False)


data = {label: load(f) for label, f, _ in MODELS}
base = data["base"]
rng = np.random.default_rng(0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1]}, layout="constrained")
x, w = np.arange(len(GROUPS)), 0.8 / len(MODELS)
for i, (label, _, color) in enumerate(MODELS):
    vals = [100 * np.mean([v for k, v in data[label].items() if ds is None or k[0] == ds]) for _, ds in GROUPS]
    xs = x - 0.4 + w * (i + 0.5)
    a1.bar(xs, vals, w * 0.92, color=color, label=label, zorder=3)
    for xx, v in zip(xs, vals):
        a1.text(xx, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=TEXT2)
a1.set_xticks(x, [g for g, _ in GROUPS], fontsize=9.5, color=TEXT)
a1.set_ylim(0, 60)
style(a1, "Accuracy with no CoT (24 samples per item)", "% correct")
for i, (label, _, color) in enumerate(MODELS[1:]):
    d = np.array([data[label][k] - base[k] for k in base])
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    a2.errorbar(i, 100 * d.mean(), yerr=[[100 * (d.mean() - lo)], [100 * (hi - d.mean())]], fmt="o", color=color, ms=9, capsize=5, lw=2, zorder=3)
    a2.text(i + 0.12, 100 * d.mean(), f"{100 * d.mean():+.1f}", va="center", fontsize=8.5, color=TEXT2)
a2.axhline(0, color=TEXT2, lw=1, ls="--", zorder=2)
a2.set_xticks(range(len(MODELS) - 1), [m[0].replace(" s", "\ns") for m in MODELS[1:]], fontsize=8.5, color=TEXT)
a2.set_ylim(-6, 6)
a2.set_xlim(-0.5, len(MODELS) - 1.5)
style(a2, "Difference from base, same items (95% CI over items)", "percentage points")
fig.legend(*a1.get_legend_handles_labels(), loc="outside upper center", ncol=len(MODELS), frameon=False, fontsize=10, labelcolor=TEXT)
fig.supxlabel("Transfer-eval items (36 GPQA, 36 HLE, 36 MMLU-Pro), eval prompt with an empty analysis channel and 'ANSWER:' prefilled, temperature 1, 24 samples per item; 0 unparsed answers.", fontsize=8.5, color=TEXT2)
fig.savefig(OUT, dpi=130, facecolor=SURFACE)
print(OUT)
