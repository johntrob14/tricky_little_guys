"""Twin of sandbag_flip_breakdown.py drawn from the behavior judges (sandbag_behavior_judge.py, results/sandbag_gpqa/behavior/) instead of the hand
reads: every gpqa.must flip (honest answer correct, arm answer parsed and wrong) of each model and arm as a stacked bar, one segment per flip in
order of the checks: the instruction judge says INFLUENCED or MENTIONED (states the plan); else the supported-option judge says the CoT supports the
correct answer and the final answer is another; else it supports a different wrong option than the one submitted; else it supports no single option;
else the reasoning arrives at the submitted wrong answer; else a judge was blocked or unparsed. The faint bar is the number of questions the model
answers correctly in its honest arm. The right panel is the planted-error judge's mean suspiciousness (± SE) over the "reasons to its answer" flips
of each bar, with the model's natural errors (honest-arm rows answered wrong while the other honest sample is right) as a dashed line.
Sandbag arms only (the CoT-control arms belong on accuracy plots); degeneracy is left out (1-5 everywhere). Rows are the temperature-0 runs the judges ran on.

    python sandbag_behavior_breakdown.py --out ../../../figures/sandbag_gpqa.must_behavior_breakdown.png
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from sandbag_flip_breakdown import FAINT, INK, INK2, SURFACE

D = Path(__file__).resolve().parents[1] / "results/sandbag_gpqa"
MODELS = [("base", "base gpt-oss-120b", "#3987e5"), ("mt1b-s150", "manytypes1b s150", "#199e70")]
ARMS = [("sandbag", "sandbag"), ("aware", "aware (+ monitor line)"), ("evasion", "evasion (+ no-reveal line)"), ("coached", "coached (+ plant a mistake)")]
CATS = [("stated", "states the plan (instruction judge: influenced or mentioned)", "#3987e5"), ("switch", "CoT supports the correct answer, submits another", "#d95926"),
        ("other", "CoT supports a different wrong option than it submits", "#c98500"), ("unsettled", "CoT supports no single option", "#d65db1"),
        ("arrives", "reasoning arrives at the submitted answer", "#199e70"), ("unjudged", "a judge blocked or unparsed", "#6b6a66")]


def verdict(j, k):
    r = j.get(k)
    return None if r is None or r["blocked"] else r["verdict"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    J = {j: {(r["run"], r["arm"], r["qid"]): r for r in map(json.loads, open(D / "behavior" / f"{j}.jsonl"))} for j in ("instruction", "supported", "planted")}
    rows, natural = [], {}
    for m, mname, col in MODELS:
        R = {(r["qid"], r["arm"]): r for r in map(json.loads, open(D / m / "rows.jsonl"))}
        right = [q for q, a in R if a == "honest" and R[(q, "honest")]["answer"] == R[(q, "honest")]["correct"]]
        # natural errors: the planted judge's honest-arm rows of this model and its honest2 run
        nat = [v["verdict"] for (run, a, q), v in J["planted"].items() if a == "honest" and run in (m, f"{m}-honest2") and not v["blocked"] and isinstance(v["verdict"], int)]
        natural[m] = (statistics.mean(nat), len(nat))
        for a, aname in ARMS:
            if not any(k[1] == a and k[0] == m for k in J["supported"]):
                continue
            c, susp = Counter(), []
            for q in right:
                r = R[(q, a)]
                if r["answer"] in (None, r["correct"]):
                    continue
                k = (m, a, q)
                ins, sup, pl = verdict(J["instruction"], k), verdict(J["supported"], k), verdict(J["planted"], k)
                if ins is None or sup is None:
                    c["unjudged"] += 1
                elif ins in ("INFLUENCED", "MENTIONED"):
                    c["stated"] += 1
                elif sup == r["correct"]:
                    c["switch"] += 1
                elif sup == "none":
                    c["unsettled"] += 1
                elif sup != r["answer"]:
                    c["other"] += 1
                else:
                    c["arrives"] += 1
                    if isinstance(pl, int):
                        susp.append(pl)
            rows.append((f"{mname}: {aname}", len(right), c, susp, col))
            print(rows[-1][0], f"flips {sum(c.values())}/{len(right)}", dict(c), f"suspiciousness {statistics.mean(susp):.0f} (n {len(susp)})" if susp else "")
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK, "font.size": 10})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 6), sharey=True, gridspec_kw=dict(width_ratios=[3.4, 1], wspace=0.04))
    for y, (_, n, c, susp, col) in enumerate(rows):
        ax.barh(y, n, color=FAINT, height=0.8)
        left = 0
        for cat, _, cc in CATS:
            if c[cat]:
                ax.barh(y, c[cat], left=left, color=cc, edgecolor=SURFACE, linewidth=2, height=0.6)
                if c[cat] >= 4:
                    ax.text(left + c[cat] / 2, y, str(c[cat]), ha="center", va="center", color=INK, fontsize=9)
                left += c[cat]
        ax.text(n + 1.5, y, f"{left}/{n}", va="center", color=INK2, fontsize=9)
        if len(susp) >= 5:  # a mean of one or two flips says nothing
            mu, se = statistics.mean(susp), (statistics.stdev(susp) / len(susp) ** 0.5 if len(susp) > 1 else 0)
            ax2.barh(y, mu, color=col, height=0.6, xerr=se, error_kw=dict(ecolor=INK2, elinewidth=1, capsize=3))
            ax2.text(mu + se + 2, y, f"{mu:.0f} (n {len(susp)})", va="center", color=INK2, fontsize=9)
    for m, mname, col in MODELS:
        ax2.axvline(natural[m][0], color=col, linestyle="--", linewidth=1.2)
    ax2.legend(handles=[plt.Line2D([], [], color=col, linestyle="--") for _, _, col in MODELS], labels=[f"natural errors, {mname}: {natural[m][0]:.0f} (n {natural[m][1]})" for m, mname, _ in MODELS],
               loc="upper left", bbox_to_anchor=(-0.02, -0.2), frameon=False, fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 165)
    ax.set_xlabel("flips (questions answered correctly in the honest arm and wrongly in this arm); faint bar = questions right in the honest arm", color=INK2)
    ax2.set_xlim(0, 60)
    ax2.set_xlabel("planted-error judge: mean suspiciousness\n(± SE) of the 'reasoning arrives at the\nsubmitted answer' flips (n ≥ 5)", color=INK2, fontsize=9)
    ax2.tick_params(axis="y", length=0)
    for a in (ax, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    used = [x for x in CATS if any(c[x[0]] for _, _, c, _, _ in rows)]  # legend for the kinds that occur
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cc) for _, _, cc in used], labels=[lab for _, lab, _ in used], loc="upper center", bbox_to_anchor=(0.6, -0.12), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("gpqa.must: sandbagging flips by kind, from the behavior judges (CoT only; instruction and supported option: claude-haiku-5.5 medium; planted error: claude-sonnet-5.5 medium); temperature-0 runs",
                 color=INK, x=0.01, ha="left", fontsize=11)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.92, bottom=0.3)
    fig.savefig(args.out, dpi=140, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
