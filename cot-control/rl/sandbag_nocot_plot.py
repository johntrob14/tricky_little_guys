"""Figure: accuracy with and without CoT on the sandbagging question sets, per model. With CoT: the honest arm of the `wmdp.can` run
(results/sandbag/<model>-sysprompt-nohint) and of the `gpqa.must` run (results/sandbag_gpqa/<model>), over finished rows. Without CoT: sandbag_nocot.py
(analysis channel forced empty, "Answer: (" prefilled), over all rows. Dotted line: chance (1/4).

    python sandbag_nocot_plot.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from hintverb_judge import wilson
from sandbag_headline import MODELS
from sandbag_variants_plot import INK, INK2, SURFACE

matplotlib.use("Agg")
RES = Path(__file__).resolve().parents[1] / "results"


def acc(rows):
    return sum(r["answer"] == r["correct"] for r in rows), len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[m for m in MODELS if (RES / "sandbag_nocot" / m).is_dir()], choices=list(MODELS), help="default: every model with a sandbag_nocot run")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[3] / "figures/sandbag_nocot.png")
    args = ap.parse_args()
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "grid.color": "#3a3a38", "hatch.color": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, ds, cot_run, title in [(axes[0], "wmdp", lambda m: RES / f"sandbag/{m}-sysprompt-nohint", "wmdp questions (150, OpenAI monitorability-evals); with CoT: wmdp.can runs"), (axes[1], "gpqa", lambda m: RES / f"sandbag_gpqa/{m}", "gpqa questions (198 GPQA diamond); with CoT: gpqa.must runs")]:
        for i, m in enumerate(args.models):
            label, col, _ = MODELS[m]
            cot = [r for r in map(json.loads, open(cot_run(m) / "rows.jsonl")) if r["arm"] == "honest" and r["finish"] != "length"]
            nocot = [r for r in map(json.loads, open(RES / f"sandbag_nocot/{m}/rows.jsonl")) if r["dataset"] == ds]
            for x, (k, n), hatch in [(i - 0.2, acc(cot), None), (i + 0.2, acc(nocot), "//")]:
                lo, hi = wilson(k, n)
                ax.bar(x, k / n, 0.38, color=col, hatch=hatch, linewidth=0)
                ax.errorbar(x, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
                ax.text(x, hi + 0.01, f"{k}/{n}", ha="center", va="bottom", fontsize=8, rotation=90, color=INK2)
        ax.axhline(0.25, color=INK2, linestyle=":", linewidth=1)
        ax.set_xticks(range(len(args.models)), [MODELS[m][0].split(" (")[0].replace(" ", "\n", 1) for m in args.models], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{title}\nsolid: with CoT (honest arm, finished rows); hatched: no CoT", fontsize=11)
        ax.grid(axis="y", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("accuracy (Wilson 95%)")
    fig.suptitle('Accuracy with and without CoT. Tinker, harmony effort medium, T 0, one sample. No CoT: analysis channel forced empty, final channel prefilled "Answer: (". Dotted: chance', fontsize=11)
    fig.subplots_adjust(top=0.82, bottom=0.12, wspace=0.12, left=0.05, right=0.99)
    fig.savefig(args.out, dpi=140, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
