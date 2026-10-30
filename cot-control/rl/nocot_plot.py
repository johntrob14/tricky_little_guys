"""Bar chart of nocot_or.py results: accuracy per dataset and overall, forced-answer no-CoT (prefill arm) vs free response with thinking
off, for the whole CoT-Control QA pool ('normal') and the Astra items, with Wilson 95% intervals and the chance level.

    python nocot_plot.py ../results/nocot/gemma4_31b/rows.jsonl ../../../figures/nocot_gemma4_31b.png --title "Gemma 4 31B"
"""
import argparse
import json
import math

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
DATASETS = ["gpqa", "hle", "mmlu_pro", "all"]
ARMS = [("prefill", "forced answer, no CoT (prefill ANSWER:)", "#3987e5"), ("free", "free response, thinking off (visible reasoning)", "#d95926")]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.rows)]
    plt.rcParams.update({"figure.facecolor": "#1a1a19", "axes.facecolor": "#1a1a19", "axes.edgecolor": "#52514e", "axes.labelcolor": "#f0efec",
                         "xtick.color": "#c3c2b7", "ytick.color": "#c3c2b7", "text.color": "#f0efec", "grid.color": "#333331", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (name, sel) in zip(axes, (("normal: all 1214 CoT-Control QA items", lambda r: True), ("astra: the 107 items of the 108-unit Astra draw", lambda r: r["astra"]))):
        for j, (arm, label, color) in enumerate(ARMS):
            for i, ds in enumerate(DATASETS):
                rs = [r for r in rows if sel(r) and r["arm"] == arm and (ds == "all" or r["dataset"] == ds)]
                if not rs:
                    continue
                k, n = sum(r["correct"] for r in rs), len(rs)
                lo, hi = wilson(k, n)
                x = i + (j - 0.5) * 0.38
                ax.bar(x, k / n, 0.34, color=color, label=label if i == 0 else None, zorder=3)
                ax.errorbar(x, k / n, yerr=[[k / n - lo], [hi - k / n]], color="#f0efec", capsize=3, lw=1, zorder=4)
                ax.text(x, hi + 0.015, f"{k}/{n}", ha="center", va="bottom", fontsize=8, color="#c3c2b7")
        for i, ds in enumerate(DATASETS):
            rs = [r for r in rows if sel(r) and r["arm"] == ARMS[0][0] and (ds == "all" or r["dataset"] == ds)]
            if rs:
                ax.hlines(sum(1 / r["n_options"] for r in rs) / len(rs), i - 0.42, i + 0.42, colors="#c3c2b7", linestyles=":", lw=1.5, zorder=5, label="chance (mean 1/options)" if i == 0 else None)
        ax.set_xticks(range(len(DATASETS)), DATASETS)
        ax.set_title(name, fontsize=11)
        ax.grid(axis="y", zorder=0)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"{args.title}: accuracy without chain of thought on the CoT-Control QA items (OpenRouter raw completions via {'+'.join(sorted({r['provider'] for r in rows}))}, thinking off, T 0, Wilson 95%)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
