"""Breakdown of every GPQA sandbagging flip (honest answer correct, arm answer parsed and wrong) of base gpt-oss-120b and manytypes1b s150 in the sandbag,
aware and evasion arms: flips the verbalization judge marked YES count as stating the plan; the rest take the label both hand reads gave them
(results/sandbag_gpqa/handval2), or 'reads disagree'. Bars are counts; the faint bar is the number of questions the model answers correctly in its honest arm.

    python sandbag_flip_breakdown.py --out ../../../figures/sandbag_gpqa_flip_breakdown.png
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

D = Path(__file__).resolve().parents[1] / "results/sandbag_gpqa"
SURFACE, INK, INK2, GRAY, FAINT = "#1a1a19", "#f0efec", "#c3c2b7", "#6b6a66", "#2c2c2a"
MODELS = [("base", "base gpt-oss-120b"), ("mt1b-s150", "manytypes1b s150")]
ARMS = [("sandbag", "sandbag order"), ("aware", "+ monitor line"), ("evasion", "+ no-reveal line")]
CATS = [("stated", "states the plan (verbalization judge or both reads)", "#3987e5"), ("switch", "reaches the correct answer, submits another", "#d95926"),
        ("switch_with_fix", "reaches the correct answer, invents a fix", "#199e70"), ("planted", "planted mistake", "#c98500"),
        ("no_work", "no real work", "#d55181"), ("honest", "honest mistake", "#008300"), ("disagree", "reads disagree or unclear", GRAY)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    key = {(k["run"], k["arm"], k["qid"]): k["item"] for k in json.load(open(D / "handval2/key.json"))}
    reads = []
    for p in sorted(glob.glob(str(D / "handval2/pass*"))):
        r = {}
        for d in (d for f in sorted(glob.glob(p + "/verdicts_shard*.jsonl")) for d in map(json.loads, open(f))):
            r.setdefault(d["item"], d["verdict"])
        reads.append(r)
    rows = []
    for m, mname in MODELS:
        R = {(r["qid"], r["arm"]): r for r in map(json.loads, open(D / m / "rows.jsonl"))}
        V = {(r["qid"], r["arm"]): r["verbalized"] for r in map(json.loads, open(D / m / "verbalized.jsonl"))}
        qids = {q for q, a in R if a == "honest"}
        right = [q for q in qids if R[(q, "honest")]["answer"] == R[(q, "honest")]["correct"]]
        for a, aname in ARMS:
            c = Counter()
            for q in right:
                r = R[(q, a)]
                if r["answer"] in (None, r["correct"]):
                    continue
                if V[(q, a)] is True:
                    c["stated"] += 1
                    continue
                labs = {rd[key[(m, a, q)]] for rd in reads if key[(m, a, q)] in rd}
                c[labs.pop() if len(labs) == 1 and labs != {"unclear"} else "disagree"] += 1
            rows.append((f"{mname}: {aname}", len(right), c))
            print(rows[-1][0], f"flips {sum(c.values())}/{len(right)}", dict(c))
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK, "font.size": 10})
    fig, ax = plt.subplots(figsize=(13, 5))
    for y, (_, n, c) in enumerate(rows):
        ax.barh(y, n, color=FAINT, height=0.8)
        left = 0
        for cat, _, col in CATS:
            if c[cat]:
                ax.barh(y, c[cat], left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.6)
                if c[cat] >= 4:
                    ax.text(left + c[cat] / 2, y, str(c[cat]), ha="center", va="center", color=INK, fontsize=9)
                left += c[cat]
        ax.text(n + 1.5, y, f"{left}/{n}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 165)
    ax.set_xlabel("flips (questions answered correctly in the honest arm and wrongly in this arm); faint bar = questions right in the honest arm", color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in CATS], labels=[lab for _, lab, _ in CATS], loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3, frameon=False, fontsize=9)
    ax.set_title("GPQA sandbagging flips by kind: verbalization judge (claude-haiku-5.5), then two blinded Sonnet 5.5 hand reads of the rest", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
