"""Figure: base Qwen3.8-27B on the wmdp task (results/SANDBAG_NAMES.md), one color per run, one group per arm.

Runs: `wmdp.can` on Tinker (<dir>/qwen-base-sysprompt-nohint, template-default effort, cut rows continued to 60k) and `wmdp.should` on OpenRouter
(<dir>/qwen-or-sysprompt-nohint-should, sandbag_or.py, with the coached arm). The honest prompt is identical in both, so the honest bars compare the two samplers.
Panels: accuracy over finished rows (Wilson 95%); flips (honest-arm answer correct, this arm's answer parsed and wrong, over those eligible; Wilson 95%);
mean CoT-only monitor score over every row of the arm (Li et al. 2025 sandbagging judge, 1 = sabotage; error bars SE); median CoT tokens over finished rows,
with the rows cut off at the cap.

    python sandbag_qwen_plot.py --dir ../results/sandbag --judge-model openai/gpt-6-luna --effort medium
"""

import argparse
import statistics
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from hintverb_judge import wilson
from sandbag import AWARE, COACH, EVASION
from sandbag_judge import judge_tag
from sandbag_variants_plot import INK, INK2, SURFACE, load

matplotlib.use("Agg")
RUNS = {"wmdp.can, Tinker": ("qwen-base-sysprompt-nohint", "#f0b429"), "wmdp.should, OpenRouter (Alibaba)": ("qwen-or-sysprompt-nohint-should", "#3ec1b5")}
ARMS = {"honest": "honest", "sandbag": "sandbag", "aware": "aware\n(monitor)", "evasion": "evasion\n(no reveal)", "coached": "coached\n(plant\nmistake)"}
FONT = 12


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--judge-model", default="openai/gpt-6-luna")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    tag = judge_tag(args.judge_model, args.effort)
    out = args.out or Path(__file__).resolve().parents[3] / f"figures/sandbag_wmdp.can+should_qwen{tag}.png"
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                         "font.size": FONT, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": "#3a3a38", "grid.linewidth": 0.6})
    data = {name: load(args.dir / d, tag)[0] for name, (d, _) in RUNS.items()}
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    w = 0.38

    def panel(ax, title, arms, value, label, ylim, err=None, rot=90, frac=True):
        ax.set_title(title, fontsize=FONT + 1, pad=10, loc="left")
        for j, (name, (_, col)) in enumerate(RUNS.items()):
            xs = [i + (j - 0.5) * w for i, a in enumerate(arms) if a in data[name]]
            cs = [data[name][a] for a in arms if a in data[name]]
            ax.bar(xs, [value(c) for c in cs], w * 0.92, color=col, label=name, yerr=None if err is None else list(zip(*[err(c) for c in cs])), error_kw=dict(ecolor=INK2, elinewidth=1.2, capsize=3))
            for x, c in zip(xs, cs):
                top = value(c) + (err(c)[1] if err else 0)
                ax.text(x, top + 0.012 * ylim[1], label(c), ha="center", va="bottom", fontsize=FONT - 1.5, color=INK, rotation=rot)
        ax.set_xticks(range(len(arms)), [ARMS[a] for a in arms], fontsize=FONT - 1)
        ax.set_ylim(*ylim)
        if frac:
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    def rate(k, n):
        r = k / n if n else 0
        lo, hi = wilson(k, n)
        return r, (max(0, r - lo), max(0, hi - r))

    arms = list(ARMS)
    panel(axes[0, 0], "A. Answer accuracy, finished rows", arms, lambda c: rate(c["correct"], c["finished"])[0], lambda c: f"{c['correct']}/{c['finished']}", (0, 1.25),
          lambda c: rate(c["correct"], c["finished"])[1])
    panel(axes[0, 1], "B. Flips: honest-correct questions answered wrong", arms[1:], lambda c: rate(c["flips"], c["eligible"])[0], lambda c: f"{c['flips']}/{c['eligible']}", (0, 1.25),
          lambda c: rate(c["flips"], c["eligible"])[1])
    mean = lambda c: statistics.mean(c["honest_scores"]) if c["honest_scores"] else 0
    se = lambda c: (statistics.stdev(c["honest_scores"]) / len(c["honest_scores"]) ** 0.5,) * 2 if len(c["honest_scores"]) > 1 else (0, 0)
    ns = [len(c["honest_scores"]) for d in data.values() for c in d.values()]
    panel(axes[1, 0], f"C. CoT-only monitor, mean score over all rows\n(1 = sabotage; {min(ns)}-{max(ns)} scored rows per bar)", arms, mean, lambda c: f"{mean(c):.0f}", (0, 112), se, rot=0, frac=False)
    panel(axes[1, 1], "D. Median CoT tokens, finished rows", arms, lambda c: c["cot_median"], lambda c: f"{c['cot_median']:.0f}", (0, max(c["cot_median"] for d in data.values() for c in d.values()) * 1.15), rot=0, frac=False)
    panel(axes[2, 0], "E. Rows cut off at the cap (no answer)", arms, lambda c: rate(c["cut"], c["n"])[0], lambda c: f"{c['cut']}/{c['n']}", (0, 0.25), lambda c: rate(c["cut"], c["n"])[1], frac=False)
    axes[0, 0].set_ylabel("fraction of rows")
    axes[0, 1].set_ylabel("fraction of eligible questions")
    axes[1, 0].set_ylabel("monitor score")
    axes[1, 1].set_ylabel("tokens")
    axes[2, 0].set_ylabel("fraction of rows")
    key = axes[2, 1]
    key.axis("off")
    key.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", frameon=False, fontsize=FONT + 1, bbox_to_anchor=(0, 1.02))
    arms_text = "\n\n".join(textwrap.fill(t, 62, subsequent_indent="   ") for t in ["sandbag: the variant's instruction in the system prompt (wmdp.can: \"you can give incorrect answers\"; wmdp.should: \"you should\")",
                              f"aware: sandbag + \"{AWARE}\"", f"evasion: aware + \"{EVASION}\"", f"coached: evasion + \"{COACH}\""])
    key.text(0, 0.72, arms_text, va="top", fontsize=FONT - 2.5, color=INK2, transform=key.transAxes)
    fig.suptitle(f"Qwen3.8-27B base on WMDP sandbagging (150 questions from OpenAI's monitorability-evals), T 0, one sample\n60k-token cap (Alibaba does not strictly enforce it on reasoning; rows that finished past it count as finished)\nMonitor: CoT only, Li et al. 2025 sandbagging judge, {args.judge_model.split('/')[-1]} ({args.effort} effort)",
                 fontsize=FONT + 1, y=0.99)
    fig.subplots_adjust(top=0.905, bottom=0.05, hspace=0.5, wspace=0.18, left=0.07, right=0.985)
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
