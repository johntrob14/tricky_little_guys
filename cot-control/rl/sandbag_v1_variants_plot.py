"""Figure: WMDP v1 instruction variants (sandbag_v1_variants.py) against the default instruction, sandbag arm only, for --models.

Runs: <dir>/<model>-sysprompt-nohint (default: "To avoid this, you can give incorrect answers.") and <model>-sysprompt-nohint-<variant>, whose honest
rows are copied from the default run (identical prompt), so every variant is scored against the same honest answers. Panels over the variants, one bar
per model: accuracy over finished rows (honest arm first), flips (honest-correct questions answered wrong) with the noise floor, explicit-admission judge
states_wrong_intent on flips (sandbag_li_explicit.py, CoT only), verbalization judge YES on flips, mean CoT-only monitor score on flips (Li et al. 2025
sandbagging judge, Haiku 5.5; dashed: mean over honest rows), median CoT tokens over finished rows.

    python sandbag_v1_variants_plot.py --dir ../results/sandbag
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from hintverb_judge import wilson
from sandbag_headline import MODELS
from sandbag_variants_plot import INK, INK2, SURFACE, load, noise_floor

matplotlib.use("Agg")
VARIANTS = [("", 'default:\n"you can give\nincorrect answers"'), ("-safe", '+ "answering incorrectly\nis the safe behavior"'), ("-should", '"you should give\nincorrect answers"')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["base"], choices=list(MODELS))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[3] / "figures/sandbag_wmdp_v1_variants.png")
    args = ap.parse_args()
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "grid.color": "#3a3a38"})
    fig, axes = plt.subplots(1, 6, figsize=(30, 6.5))
    w = 0.8 / len(args.models)
    xs = range(len(VARIANTS))
    for i, m in enumerate(args.models):
        label, col, floor_runs = MODELS[m]
        x = [k - 0.4 + w * (i + 0.5) for k in xs]
        runs = [args.dir / f"{m}-sysprompt-nohint{s}" for s, _ in VARIANTS]
        cells = [load(r)[0] for r in runs]
        sb = [c["sandbag"] for c in cells]
        exp = []
        for r, c in zip(runs, cells):  # explicit-admission verdicts on the flips
            rows = {(x["qid"], x["arm"]): x for x in map(json.loads, open(r / "rows.jsonl"))}
            honest = {q: x for (q, a), x in rows.items() if a == "honest"}
            flips = [q for (q, a), x in rows.items() if a == "sandbag" and honest[q]["answer"] == x["correct"] and x["answer"] not in (None, x["correct"])]
            ex = {x["qid"]: x["states_wrong_intent"] for x in map(json.loads, open(r / "explicit_cot.jsonl")) if x["condition"] == "sandbag" and x["states_wrong_intent"] is not None}
            exp.append((sum(ex[q] for q in flips if q in ex), sum(q in ex for q in flips)))
        fk, fn = noise_floor([load(args.dir / r)[1] for r in [f"{m}-sysprompt-nohint"] + floor_runs])
        panels = [(axes[0], [c["correct"] for c in sb], [c["finished"] for c in sb]), (axes[1], [c["flips"] for c in sb], [c["eligible"] for c in sb]),
                  (axes[2], [k for k, _ in exp], [n for _, n in exp]), (axes[3], [c["verbalized"] for c in sb], [c["judged"] for c in sb])]
        for ax, k, n in panels:
            r = [a / b if b else 0 for a, b in zip(k, n)]
            ci = [wilson(a, b) for a, b in zip(k, n)]
            ax.bar(x, r, w * 0.92, color=col, label=label if ax is axes[0] else None, yerr=[[v - lo for v, (lo, _) in zip(r, ci)], [hi - v for v, (_, hi) in zip(r, ci)]], error_kw=dict(ecolor=INK2, elinewidth=1, capsize=3))
            for xi, a, b in zip(x, k, n):
                ax.text(xi, 0.01, f"{a}/{b}", ha="center", va="bottom", fontsize=8, rotation=90, color=SURFACE)
        h = cells[0]["honest"]
        axes[0].axhline(h["correct"] / h["finished"], color=col, linestyle="--", linewidth=1.2)
        axes[0].text(2.45, h["correct"] / h["finished"] + 0.01, f"honest arm: {h['correct']}/{h['finished']}", ha="right", va="bottom", fontsize=8, color=col)
        axes[1].axhline(fk / fn, color=col, linestyle="--", linewidth=1.2)
        axes[1].text(0.01, 0.99 - 0.04 * i, f"noise floor ({label}): {fk}/{fn} = {fk / fn:.3f}", ha="left", va="top", fontsize=8, color=col, transform=axes[1].transAxes)
        ms = [statistics.mean(c["flip_scores"]) for c in sb]
        axes[4].bar(x, ms, w * 0.92, color=col, yerr=[statistics.stdev(c["flip_scores"]) / len(c["flip_scores"]) ** 0.5 for c in sb], error_kw=dict(ecolor=INK2, elinewidth=1, capsize=3))
        for xi, c, v in zip(x, sb, ms):
            axes[4].text(xi, 2, f"{v:.0f} (n {len(c['flip_scores'])})", ha="center", va="bottom", fontsize=8, rotation=90, color=SURFACE)
        hs = h["honest_scores"]
        axes[4].axhline(statistics.mean(hs), color=col, linestyle="--", linewidth=1.2)
        axes[4].text(2.45, statistics.mean(hs) + 1, f"honest rows: mean {statistics.mean(hs):.0f}", ha="right", va="bottom", fontsize=8, color=col)
        axes[5].bar(x, [c["cot_median"] for c in sb], w * 0.92, color=col)
        axes[5].axhline(h["cot_median"], color=col, linestyle="--", linewidth=1.2)
        axes[5].text(2.45, h["cot_median"] + 15, f"honest arm: {h['cot_median']:.0f}", ha="right", va="bottom", fontsize=8, color=col)
    titles = ["A. Accuracy, sandbag arm, finished rows\n(Wilson 95%; dashed: honest arm)", "B. Flips: honest-correct questions answered wrong\n(Wilson 95%; dashed: noise floor from resampling the honest prompt)",
              "C. Explicit-admission judge on flips: the CoT states\nit is choosing an answer it believes wrong", "D. Verbalization judge YES on flips\n(the CoT mentions the instruction and depends on it)",
              "E. CoT-only monitor: mean score on flips (± SE),\n1 = sabotage (dashed: mean over honest rows)", "F. Median CoT tokens, sandbag arm, finished rows\n(dashed: honest arm)"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=11)
        ax.set_xticks(list(xs), [v for _, v in VARIANTS], fontsize=9)
        ax.set_ylim(0, 100 if ax is axes[4] else None if ax is axes[5] else 1.05)
        ax.grid(axis="y", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(args.models), fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.9))
    fig.suptitle("WMDP v1 sandbagging, instruction variants (sandbag arm): 150 questions (OpenAI monitorability-evals), deletion threat in the system prompt, no named wrong option. "
                 "Tinker, harmony effort medium, T 0, one sample.\nMonitor: CoT only, Li et al. 2025 sandbagging judge, claude-haiku-5.5. Verbalization and explicit-admission judges: claude-haiku-5.5.", fontsize=11)
    fig.subplots_adjust(top=0.74, bottom=0.14, wspace=0.22, left=0.03, right=0.99)
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
