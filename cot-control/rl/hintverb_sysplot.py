"""Oversight lines in the system prompt (hintverb.py --placement system) vs no line, per model, pooled over the three Chen et al. 2025 hints: the share of
pairs switched to the hint and the judge's verbalized share of switched rows (Wilson 95%; white ticks: length-adjusted, hintverb_pressure.length_adjust),
and the mean CoT tokens of switched rows. Reads the summary hintverb_pressure.py writes for these arms.

    python hintverb_pressure.py --dir ../results/hintverb --models base grpo3-s100 mt1b-s150 --suffixes none cot_private@sys user_reads@sys answer_checked@sys auto_monitor@sys --summary pressure_summary_sys.json --fig ../../../figures/hintverb_pressure_sys_byhint.png
    python hintverb_sysplot.py --summary ../results/hintverb/pressure_summary_sys.json --out ../../../figures/hintverb_pressure_sys.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from hintverb import SUFFIXES
from hintverb_judge import wilson
from hintverb_pressure import LABELS
from sandbag_headline import MODELS
from sandbag_variants_plot import INK, INK2, SURFACE

matplotlib.use("Agg")
ORDER = ["none", "neutral", "cot_private", "user_reads", "user_reads_hint", "answer_checked", "grader", "auto_monitor", "monitor_discard"]  # controls first, then the lines claiming a check


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    S = json.load(open(args.summary))
    models = [m for m in S if m in MODELS]
    lines = list(S[models[0]].keys() - {"length_fit"})
    lines = sorted(lines, key=lambda s: ORDER.index(s.removesuffix("@sys")))
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "grid.color": "#3a3a38"})
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        x = [k - 0.4 + w * (i + 0.5) for k in range(len(lines))]
        c = [S[m][s]["hints"]["pooled"] for s in lines]
        panels = [(axes[0], [x["p"] for x in c], [wilson(x["to_hint"], x["pairs"]) for x in c], [x["p_lenadj"] for x in c], [f"{x['to_hint']}/{x['pairs']}" for x in c]),
                  (axes[1], [x["faithfulness"] for x in c], [x["faithfulness_ci"] for x in c], [x["faithfulness_lenadj"] for x in c], [f"{x['verbalized']}/{x['judged']}" for x in c]),
                  (axes[2], [x["cot_tokens_switched_mean"] for x in c], None, None, [f"{x['cot_tokens_switched_mean']:.0f}" for x in c])]
        for ax, v, ci, adj, lab in panels:
            err = None if ci is None else [[vi - lo for vi, (lo, hi) in zip(v, ci)], [hi - vi for vi, (lo, hi) in zip(v, ci)]]
            ax.bar(x, v, w * 0.92, yerr=err, color=MODELS[m][1], ecolor=INK2, capsize=3, label=MODELS[m][0], zorder=2)
            if adj is not None:
                ax.scatter(x, adj, marker="_", s=150, linewidths=2, color="white", zorder=3, label="length-adjusted" if i == 0 else None)
            for k, (xi, vi, t) in enumerate(zip(x, v, lab)):  # above the error bar and the length-adjusted tick
                ax.text(xi, (vi + 15) if ci is None else max(ci[k][1], adj[k]) + 0.02, t, ha="center", va="bottom", rotation=90, fontsize=7.5, color=INK2, zorder=4)
    titles = ["Answers switched to the hinted letter, among pairs\n(unhinted answer differed from the hint; Wilson 95%)",
              f"Judge: CoT verbalizes the hint, among answers switched to it\n({S['judge']['model'].split('/')[-1]}, effort {S['judge']['effort']}; Wilson 95%)", "Mean CoT tokens of the answers switched to the hint"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=11, loc="left")
        ax.set_xticks(range(len(lines)), [LABELS[s.removesuffix("@sys")] for s in lines], fontsize=9)
        ax.grid(axis="y", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:2]:
        ax.set_ylim(0, 1.17)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[2].set_ylim(0, axes[2].get_ylim()[1] * 1.12)
    h, l = axes[0].get_legend_handles_labels()
    h, l = h[1:] + h[:1], l[1:] + l[:1]  # models first, the length-adjusted tick last
    fig.legend(h, l, loc="upper center", ncol=len(models) + 1, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.93))
    n = S[models[0]][lines[0]]["n_questions"]
    fig.suptitle(f"Hint verbalization with an oversight line added as a second paragraph of the system prompt (pooled over the metadata, grader and unethical hints of Chen et al. 2025; {n} questions; gpt-oss-120b harmony effort medium, T 0, Tinker)", fontsize=12, y=0.99)
    fig.text(0.01, 0.01, "   ".join(f"{LABELS[s.removesuffix('@sys')].replace(chr(10), ' ')}: \"{SUFFIXES[s.removesuffix('@sys')]}\"" for s in lines if s != "none"), fontsize=8.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
