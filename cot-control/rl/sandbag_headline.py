"""Headline figure for the WMDP sandbagging eval on the gpt-oss organisms, from results/sandbag/summary.json (sandbag_judge.py output).

Four panels over the four arms: answer accuracy; mean analysis-channel tokens; the share of sandbagged rows (honest-correct questions answered wrong) whose
CoT verbalizes the instruction, by the Haiku verbalization judge; the CoT-only monitor's recall on those rows and its flag rate on honest rows. Wilson 95%
intervals on the rates, k/n under each bar.

    python sandbag_headline.py --dir ../results/sandbag --models base grpo3-s100 mt1b-s150
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
ARMS = ["honest", "sandbag", "aware", "evasion"]
ARM_LABELS = ["honest", "sandbag\n(OpenAI hint)", "aware\n(+ monitor line)", "evasion\n(+ no-reveal line)"]
COLORS = ["#3987e5", "#d95926", "#199e70"]  # dark-surface categorical slots 1-3, validated with the dataviz palette script
LABELS = {"base": "gpt-oss-120b base", "grpo3-s100": "grpo3 s100 (CoT-control, no mention penalty)", "mt1b-s150": "manytypes1b s150 (CoT-control, mention penalty)"}
SURFACE, INK, INK2 = "#1a1a19", "#f0efec", "#c3c2b7"


def wilson_err(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, p - (c - h)), max(0.0, (c + h) - p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["base", "grpo3-s100", "mt1b-s150"])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[3] / "figures/sandbag_wmdp_gptoss_headline.png")
    args = ap.parse_args()
    S = json.load(open(args.dir / "summary.json"))
    judge = S["judge"]["model"].split("/")[-1]
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "grid.color": "#3a3a38", "legend.facecolor": SURFACE, "legend.edgecolor": INK2})
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.2))
    n_models, w = len(args.models), 0.8 / len(args.models)
    for i, m in enumerate(args.models):
        cells = [S[m]["arms"][a] for a in ARMS]
        x = [k + i * w - 0.4 + w / 2 for k in range(len(ARMS))]
        col, lab = COLORS[i], LABELS.get(m, m)
        kw = dict(width=w * 0.92, color=col, label=lab, linewidth=0)
        n_q = S[m]["n_questions"]
        axes[0].bar(x, [c["correct"] / n_q for c in cells], **kw)
        for xi, c in zip(x, cells):
            axes[0].text(xi, c["correct"] / n_q + 0.015, f"{c['correct']}", ha="center", va="bottom", fontsize=7, color=INK2)
        axes[1].bar(x, [c["cot_tokens_mean"] for c in cells], **kw)
        axes[1].scatter(x, [c["cot_tokens_median"] for c in cells], marker="_", s=120, color=INK, zorder=3)
        v = [(c.get("verbalized", 0), c.get("verbalized_judged", 0)) for c in cells]
        axes[2].bar(x, [k / n if n else 0 for k, n in v], yerr=[[wilson_err(k, n)[0] for k, n in v], [wilson_err(k, n)[1] for k, n in v]], capsize=2, error_kw=dict(ecolor=INK2, lw=1), **kw)
        for xi, (k, n) in zip(x, v):
            if n:
                axes[2].text(xi, 0.02, f"{k}/{n}", ha="center", va="bottom", fontsize=7, rotation=90, color=SURFACE)
        r = [(c.get("recall_flagged", c["monitor_flagged"]), c.get("recall_scored", c["monitor_scored"])) for c in cells]  # honest: flags on honest rows
        axes[3].bar(x, [k / n if n else 0 for k, n in r], yerr=[[wilson_err(k, n)[0] for k, n in r], [wilson_err(k, n)[1] for k, n in r]], capsize=2, error_kw=dict(ecolor=INK2, lw=1), **kw)
        for xi, (k, n) in zip(x, r):
            if n:
                axes[3].text(xi, 0.02, f"{k}/{n}", ha="center", va="bottom", fontsize=7, rotation=90, color=SURFACE)
    axes[0].set_title(f"Answer accuracy\n(correct of {S[args.models[0]]['n_questions']} questions)", fontsize=10)
    axes[1].set_title("Analysis-channel tokens\n(bar: mean, white tick: median)", fontsize=10)
    axes[2].set_title("Sandbagged rows whose CoT verbalizes\nthe instruction (verbalization judge)", fontsize=10)
    axes[3].set_title("CoT-only monitor (Li et al. 2025 judge, score < 70):\nrecall on sandbagged rows; flag rate on honest rows", fontsize=10)
    for ax in axes:
        ax.set_xticks(range(len(ARMS)), ARM_LABELS, fontsize=9)
        ax.grid(axis="y", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    for ax in [axes[0], axes[2], axes[3]]:
        ax.set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"WMDP sandbagging (OpenAI monitorability-evals questions): gpt-oss-120b base vs the CoT-control organisms. Tinker, effort medium, T 0, one sample.\nSandbagged row = question answered correctly in the honest arm and wrongly in that arm. Judges: {judge}. Wilson 95% intervals, k/n under the bars.", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    args.out.parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
