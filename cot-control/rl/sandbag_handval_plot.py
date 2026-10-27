"""Chart and tally of the blinded GPQA hand read with reference CoTs (results/sandbag_gpqa/handval2, two independent reads per item): per source, the
reader verdict and how the CoT's own conclusion relates to the correct and submitted letters (bar = mean fraction over the passes, label = count in each
pass), and the mean deliberate score. Prints the per-source counts and the agreement between passes.

    python sandbag_handval_plot.py --dir ../results/sandbag_gpqa/handval2 --out ../../../figures/sandbag_gpqa_handval2.png
"""
import argparse
import glob
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRAY = "#1a1a19", "#f0efec", "#c3c2b7", "#6b6a66"
GROUPS = [
    ("mt1b: sandbag order", lambda k: k["source"] == "flip_not_verbalized" and k["run"] == "mt1b-s150" and k["arm"] == "sandbag"),
    ("mt1b: + monitor line", lambda k: k["source"] == "flip_not_verbalized" and k["run"] == "mt1b-s150" and k["arm"] == "aware"),
    ("mt1b: + no-reveal line", lambda k: k["source"] == "flip_not_verbalized" and k["run"] == "mt1b-s150" and k["arm"] == "evasion"),
    ("control: mt1b CoT-control arm,\nno sandbag order", lambda k: k["source"] == "control_cotctl"),
    ("control: mt1b honest\nresampling errors", lambda k: k["source"] == "control_resample"),
    ("control: verbalized flips", lambda k: k["source"] == "flip_verbalized"),
    ("other models' non-verbalized flips", lambda k: k["source"] == "flip_not_verbalized" and k["run"] != "mt1b-s150"),
]
VERDICT = [("stated", "states the plan", "#3987e5"), ("switch", "reaches the correct answer, submits another", "#d95926"),
           ("switch_with_fix", "reaches the correct answer, invents a fix", "#199e70"), ("planted", "planted mistake", "#c98500"),
           ("no_work", "no real work", "#d55181"), ("unclear", "unclear", GRAY), ("honest", "honest mistake", "#008300")]
CONCL = [("correct", "concludes the correct option", "#d95926"), ("submitted", "argues for the wrong option it submits", "#008300"),
         ("other", "concludes a third option", "#c98500"), ("none", "names no conclusion", GRAY)]


def conclusion(v, q):
    c = (v["cot_conclusion"] or "").strip().upper()[:1]
    return "correct" if c == q["correct"] else "submitted" if c == q["submitted"] else "other" if c in ("A", "B", "C", "D") else "none"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    key = {k["item"]: k for k in json.load(open(args.dir / "key.json"))}
    items = {d["item"]: d for d in map(json.loads, open(args.dir / "items.jsonl"))}
    unreadable = json.load(open(args.dir / "unreadable.json"))
    items = {i: d for i, d in items.items() if i not in unreadable}
    passes = sorted(glob.glob(str(args.dir / "pass*")))
    reads = [{} for _ in passes]  # an item read twice in a pass keeps its first read in sorted file order
    for r, p in zip(reads, passes):
        for d in (d for f in sorted(glob.glob(p + "/verdicts_shard*.jsonl")) for d in map(json.loads, open(f))):
            r.setdefault(d["item"], d)
    haiku = [{d["item"] for f in glob.glob(p + "/verdicts_shard_haiku*.jsonl") for d in map(json.loads, open(f))} for p in passes]
    for p, r, h in zip(passes, reads, haiku):
        print(f"{p}: {len(set(r) & set(items))}/{len(items)} items read ({len(h)} by Haiku 5.5: {sorted(h)}); unreadable, excluded: {sorted(unreadable)}")
    rows = []
    for name, pred in GROUPS:
        its = [i for i in items if pred(key[i])]
        verd = [Counter(r[i]["verdict"] for i in its if i in r) for r in reads]
        concl = [Counter(conclusion(r[i], items[i]) for i in its if i in r) for r in reads]
        scores = [r[i]["deliberate_score"] for r in reads for i in its if i in r]
        bothread = [i for i in its if all(i in r for r in reads)]
        agree = sum(reads[0][i]["verdict"] == reads[1][i]["verdict"] for i in bothread)
        rows.append((name, len(its), verd, concl, scores, agree))
        print(f"{name.replace(chr(10), ' ')} (n={len(its)}): verdict agreement between passes {agree}/{len(bothread)}, deliberate score mean {statistics.mean(scores):.0f}")
        for p, (v, c) in enumerate(zip(verd, concl)):
            print(f"  pass{p} ({sum(v.values())} read) verdicts {dict(v)}  conclusion {dict(c)}")
    bothall = [i for i in items if all(i in r for r in reads)]
    both = lambda lab: sum(reads[0][i]["verdict"] == lab and reads[1][i]["verdict"] == lab for i in bothall)
    either = lambda lab: sum(reads[0][i]["verdict"] == lab or reads[1][i]["verdict"] == lab for i in bothall)
    print("per-label agreement (both passes / either pass):", {lab: f"{both(lab)}/{either(lab)}" for lab, _, _ in VERDICT})
    for name, pred in GROUPS:
        its = [i for i in bothall if pred(key[i])]
        wrongarg = [i for i in its if all(conclusion(r[i], items[i]) == "submitted" for r in reads)]
        print(f"{name.replace(chr(10), ' ')}: planted in both reads {sum(all(r[i]['verdict'] == 'planted' for r in reads) for i in its)}/{len(its)}; "
              f"among items both reads say argue for the submitted wrong option: planted in both {sum(all(r[i]['verdict'] == 'planted' for r in reads) for i in wrongarg)}/{len(wrongarg)}, "
              f"planted in either {sum(any(r[i]['verdict'] == 'planted' for r in reads) for i in wrongarg)}/{len(wrongarg)}, "
              f"mean deliberate score {statistics.mean([r[i]['deliberate_score'] for r in reads for i in wrongarg]) if wrongarg else float('nan'):.0f}")
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK, "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), sharey=True, gridspec_kw={"width_ratios": [3, 3, 1.2]})
    for ax, cats, idx, title in [(axes[0], VERDICT, 2, "Reader's verdict"), (axes[1], CONCL, 3, "What the CoT itself concludes")]:
        for y, row in enumerate(rows):
            per_pass, left = row[idx], 0
            for cat, _, col in cats:
                cs = [c[cat] for c in per_pass]
                w = statistics.mean(c[cat] / sum(c.values()) for c in per_pass)
                ax.barh(y, w, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.7)
                if w >= 0.04:
                    ax.text(left + w / 2, y, "/".join(map(str, cs)) if len(set(cs)) > 1 else str(cs[0]), ha="center", va="center", color=INK, fontsize=8)
                left += w
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of items (mean over the two reads); labels: count in read 1 / read 2", color=INK2)
        ax.set_title(title, color=INK, loc="left")
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in cats], labels=[lab for _, lab, _ in cats], loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=9)
    ax = axes[2]
    for y, row in enumerate(rows):
        m = statistics.mean(row[4])
        ax.barh(y, m, color="#3987e5", height=0.7)
        ax.text(m + 2, y, f"{m:.0f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("mean over items and reads", color=INK2)
    ax.set_title("Deliberate score (0-100)", color=INK, loc="left")
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(range(len(rows)), [f"{name} (n={n})" for name, n, *_ in rows])
    axes[0].invert_yaxis()
    fig.suptitle("Blinded hand read of manytypes1b s150's GPQA flips the verbalization judge (claude-haiku-5.5) did not mark as stating the plan, with controls; "
                 "two independent Sonnet 5.5 reads per item, reader shown the model's own correct CoT on the same question", color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
