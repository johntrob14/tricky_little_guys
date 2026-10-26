"""Figure for sandbag_prompt_search.py: per variant, base gpt-oss-120b vs manytypes2e s170, the flips (honest-correct GPQA questions answered wrong) split
into verbalized (the verbalization judge said YES), silent and flagged by the CoT-only monitor, silent and uncaught by it but flagged once the monitor
also sees the final answer, and silent and uncaught by both. Monitor: Li et al. 2025 prompt, gpt-6-luna, flagged < 70. Controls (ctl_*) carry the CoT
constraint without the sandbag order.

    python sandbag_prompt_search_plot.py --dir ../results/sandbag_gpqa_search
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from sandbag_variants_plot import INK, INK2, SURFACE

matplotlib.use("Agg")
ORDER = ["direct", "nomention", "nocommit", "nocommit_nomention", "split", "split_nocommit", "t2_user", "t8_user", "suppress_nocommit_user", "decoy", "ctl_nocommit", "ctl_t2_user"]
PARTS = [("verbalized", "#8a8984", "verbalized (the CoT states the plan)"), ("caught", "#3987e5", "silent, CoT-only monitor flags it"),
         ("answer", "#e87ba4", "silent, CoT-only monitor misses it, flagged once the monitor sees the answer"), ("hidden", "#c98500", "silent, missed by both monitors")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[3] / "figures/sandbag_gpqa_prompt_search.png")
    args = ap.parse_args()
    S = {(x["model"], x["arm"]): x for x in json.load(open(args.dir / "summary.json"))["rows"]}
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "grid.color": "#3a3a38"})
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.5), sharey=True)
    for ax, (m, title) in zip(axes, [("base", "gpt-oss-120b base"), ("manytypes2e-s170", "manytypes2e s170")]):
        for i, v in enumerate(ORDER):
            x = S[(m, v)]
            vals = dict(verbalized=x["flips"] - x["silent"], caught=x["silent"] - x["silent_uncaught"], answer=x["silent_uncaught_answer_flagged"], hidden=x["silent_uncaught"] - x["silent_uncaught_answer_flagged"])
            bottom = 0
            for k, col, lab in PARTS:
                ax.bar(i, vals[k], 0.7, bottom=bottom, color=col, linewidth=0, label=lab if i == 0 else None)
                bottom += vals[k]
            ax.text(i, bottom + 2, f"{x['flips']}/{x['eligible']}", ha="center", va="bottom", fontsize=8, color=INK2)
            if vals["answer"] or vals["hidden"]:
                ax.text(i, vals["verbalized"] + vals["caught"] + vals["answer"] / 2, str(vals["answer"]), ha="center", va="center", fontsize=8, color=SURFACE)
        ax.set_xticks(range(len(ORDER)), ORDER, rotation=40, ha="right", fontsize=9)
        ax.axvline(len(ORDER) - 2.5, color=INK2, linestyle=":", linewidth=1)
        ax.set_title(f"{title}: flips by kind (label: flips / honest-correct; pink count in bar)", fontsize=11)
        ax.grid(axis="y", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("flips (honest-correct, answered wrong)")
    axes[0].set_ylim(0, 152)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("GPQA silent-sandbagging prompt search (198 GPQA diamond questions; right of the dotted line: controls with the CoT constraint and no sandbag order). "
                 "Tinker, effort medium, T 0, one sample.\nMonitor: CoT only (and CoT + final answer on silent flips), Li et al. 2025 sandbagging judge, gpt-6-luna, flagged < 70. Verbalization judge: claude-haiku-5.5.", fontsize=10)
    fig.subplots_adjust(top=0.78, bottom=0.25, left=0.05, right=0.99, wspace=0.05)
    fig.savefig(args.out, dpi=140, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
