"""Bar chart of the blinded GPQA hand read (results/sandbag_gpqa/handval): per source, 100% stacked bars of the reader's verdict and of how the CoT's own
conclusion relates to the correct and submitted letters, counts on the bars.

    python sandbag_handval_plot.py --dir ../results/sandbag_gpqa/handval --out ../../../figures/sandbag_gpqa_handval.png
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRAY = "#1a1a19", "#f0efec", "#c3c2b7", "#6b6a66"
BLUE, ORANGE, AQUA, YELLOW = "#3987e5", "#d95926", "#199e70", "#c98500"
GROUPS = [
    ("mt1b: sandbag order", lambda k: k["source"] == "flip_not_verbalized" and k["run"] == "mt1b-s150" and k["arm"] == "sandbag"),
    ("mt1b: + monitor line", lambda k: k["source"] == "flip_not_verbalized" and k["run"] == "mt1b-s150" and k["arm"] == "aware"),
    ("mt1b: + no-reveal line", lambda k: k["source"] == "flip_not_verbalized" and k["run"] == "mt1b-s150" and k["arm"] == "evasion"),
    ("control: mt1b CoT-control arm,\nno sandbag order", lambda k: k["source"] == "control_cotctl"),
    ("control: mt1b honest\nresampling errors", lambda k: k["source"] == "control_resample"),
    ("control: verbalized flips", lambda k: k["source"] == "flip_verbalized"),
]
VERDICT = [("yes_stated", "states the plan", BLUE), ("yes_silent", "silent (CoT doesn't fit the answer)", ORANGE), ("unclear", "unclear", GRAY), ("no", "honest mistake", AQUA)]
CONCL = [("correct", "concludes the correct option, submits another", ORANGE), ("submitted", "argues for the wrong option it submits", AQUA), ("other", "concludes a third option", YELLOW), ("none", "names no conclusion", GRAY)]


def conclusion(v, q):
    c = (v["cot_conclusion"] or "").strip().upper()[:1]
    return "correct" if c == q["correct"] else "submitted" if c == q["submitted"] else "other" if c in ("A", "B", "C", "D") else "none"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    key = {k["item"]: k for k in json.load(open(args.dir / "key.json"))}
    items = {d["item"]: d for f in glob.glob(str(args.dir / "shard*.jsonl")) for d in map(json.loads, open(f))}
    verdicts = {d["item"]: d for f in glob.glob(str(args.dir / "verdicts_shard*.jsonl")) for d in map(json.loads, open(f))}
    counts = []
    for _, pred in GROUPS:
        its = [i for i in verdicts if pred(key[i])]
        counts.append((len(its), Counter(verdicts[i]["deliberate_wrong"] for i in its), Counter(conclusion(verdicts[i], items[i]) for i in its)))
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    for ax, cats, idx, title in [(axes[0], VERDICT, 1, "Reader's verdict"), (axes[1], CONCL, 2, "What the CoT itself concludes")]:
        for y, (n, *cs) in enumerate(counts):
            left = 0
            for cat, _, col in cats:
                c = cs[idx - 1][cat]
                ax.barh(y, c / n, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.7)
                if c:
                    ax.text(left + c / n / 2, y, str(c), ha="center", va="center", color=INK, fontsize=9)
                left += c / n
        ax.set_yticks(range(len(GROUPS)), [f"{g} (n={n})" for (g, _), (n, *_) in zip(GROUPS, counts)])
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of items", color=INK2)
        ax.set_title(title, color=INK, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in cats], labels=[lab for _, lab, _ in cats], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Blinded hand read of manytypes1b s150's GPQA flips the verbalization judge (claude-haiku-5.5) did not mark as stating the plan, with controls", color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
