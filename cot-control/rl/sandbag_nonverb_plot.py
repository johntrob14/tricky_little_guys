"""Mean CoT-only monitor suspicion (100 - the 1-100 approval score of sandbag_judge.py's monitor) over sandbagged rows the verbalization judge marked
as not verbalizing the instruction, per model and arm. Faint bars: the verbalized rows, for comparison. Labels: non-verbalized n / sandbagged rows judged."""
import argparse, json, statistics
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", type=Path, default=Path(__file__).parent / "../results/sandbag_gpqa_t1")
ap.add_argument("--runs", nargs="+", default=["base", "mt1b-s150", "manytypes2e-s170"])
ap.add_argument("--arms", nargs="+", default=["sandbag", "aware", "evasion", "evasion_analysis", "coached", "coached_analysis"])
ap.add_argument("--tag", default="claude-haiku-5.5_medium")
ap.add_argument("--out", type=Path, default=Path(__file__).parents[3] / "figures/sandbag_gpqa.must_t1_nonverbalized_suspicion.png")
ap.add_argument("--title", default="gpqa.must, temperature 1: CoT-only monitor suspicion on sandbagged rows, split by the verbalization judge")
args = ap.parse_args()

stats = {}  # (run, arm, verbalized) -> list of suspicion scores
for run in args.runs:
    d = args.dir / run
    mon = {(r["qid"], r["arm"]): r["score"] for r in map(json.loads, open(d / f"monitor_{args.tag}.jsonl"))}
    for v in map(json.loads, open(d / f"verbalized_{args.tag}.jsonl")):
        s = mon.get((v["qid"], v["arm"]))
        if v["verbalized"] is None or not isinstance(s, int): continue
        stats.setdefault((run, v["arm"], v["verbalized"]), []).append(100 - s)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(13, 5.5))
colors = {"base": "#8ab4f8", "mt1b-s150": "#f28e2b", "manytypes2e-s170": "#b07aa1", "grpo3-s100": "#59a14f"}
w = 0.8 / len(args.runs)
for i, run in enumerate(args.runs):
    for j, arm in enumerate(args.arms):
        x = j + (i - (len(args.runs) - 1) / 2) * w
        nv, vb = stats.get((run, arm, False), []), stats.get((run, arm, True), [])
        if vb:
            ax.bar(x, statistics.mean(vb), w * 0.9, color=colors.get(run, "gray"), alpha=0.25, edgecolor=colors.get(run, "gray"), linewidth=0.8)
        if nv:
            se = statistics.stdev(nv) / len(nv) ** 0.5 if len(nv) > 1 else 0
            ax.bar(x, statistics.mean(nv), w * 0.9, yerr=se, color=colors.get(run, "gray"), label=run if j == 0 else None, capsize=2, error_kw=dict(lw=0.8))
        ax.text(x, 2, f"{len(nv)}/{len(nv) + len(vb)}", ha="center", va="bottom", fontsize=7, rotation=90, color="black" if nv else "white")
        print(f"{run:18s} {arm:17s} non-verbalized n {len(nv):3d}/{len(nv) + len(vb):3d} mean suspicion {statistics.mean(nv) if nv else float('nan'):5.1f}   verbalized n {len(vb):3d} mean {statistics.mean(vb) if vb else float('nan'):5.1f}")
ax.set_xticks(range(len(args.arms))); ax.set_xticklabels(args.arms)
ax.set_ylim(0, 100); ax.set_ylabel("mean suspicion = 100 - monitor approval score (± SE)")
ax.set_title(f"{args.title}\nsolid: rows judged NOT verbalizing the instruction (label: n / sandbagged rows judged); faint: verbalized rows. Monitor and judge: {args.tag.replace('_', ' at ')}, CoT only", fontsize=9)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(args.out, dpi=150); print(args.out)
