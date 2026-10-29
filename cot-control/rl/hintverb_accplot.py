"""Accuracy per model and system-prompt oversight line (hintverb.py arms), on the unhinted prompts and on the hinted prompts (three hints pooled; the hint
points to the correct answer on 115/479 questions), Wilson 95%. Letters parsed with the current hintverb.parse_letter; an unparsed answer counts as wrong.

    python hintverb_accplot.py --dir ../results/hintverb --out ../../../figures/hintverb_pressure_sys_accuracy.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from hintverb import parse_letter
from hintverb_judge import wilson
from hintverb_pressure import LABELS
from sandbag_headline import MODELS
from sandbag_variants_plot import INK, INK2, SURFACE

matplotlib.use("Agg")
LINES = ["none", "cot_private@sys", "user_reads@sys", "answer_checked@sys", "auto_monitor@sys"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["base", "grpo3-s100", "mt1b-s150"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "grid.color": "#3a3a38"})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    w = 0.8 / len(args.models)
    for i, m in enumerate(args.models):
        x = [k - 0.4 + w * (i + 0.5) for k in range(len(LINES))]
        for ax, hinted in zip(axes, [False, True]):
            kn = []
            for s in LINES:
                rows = [r for r in map(json.loads, open(args.dir / (m if s == "none" else f"{m}.{s}") / "rows.jsonl")) if (r["hint_type"] != "none") == hinted]
                kn.append((sum(parse_letter(r["response"]) == r["correct"] for r in rows), len(rows)))
            v, ci = [k / n for k, n in kn], [wilson(k, n) for k, n in kn]
            ax.bar(x, v, w * 0.92, yerr=[[a - lo for a, (lo, hi) in zip(v, ci)], [hi - a for a, (lo, hi) in zip(v, ci)]], color=MODELS[m][1], ecolor=INK2, capsize=3, label=MODELS[m][0], zorder=2)
            for xi, (k, n), (lo, hi) in zip(x, kn, ci):
                ax.text(xi, hi + 0.02, f"{k}/{n}", ha="center", va="bottom", rotation=90, fontsize=7.5, color=INK2)
            print(m, "hinted" if hinted else "unhinted", " ".join(f"{s}={k}/{n}" for s, (k, n) in zip(LINES, kn)))
    for ax, t in zip(axes, ["Accuracy without the hint (479 questions)", "Accuracy with a hint (three hints pooled; the hint is correct on 115/479)"]):
        ax.set_title(t, fontsize=11, loc="left")
        ax.set_xticks(range(len(LINES)), [LABELS[s.removesuffix("@sys")] for s in LINES], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(args.models), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Accuracy with an oversight line in the system prompt (MMLU + GPQA Diamond; gpt-oss-120b harmony effort medium, T 0, Tinker; Wilson 95%)", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
