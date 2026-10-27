"""Per-item CoT-control compliance against a CoT-uplift measure on the 360 full-GPQA items (results/transfer/gpqafull-<model>[-seed1], results/difficulty/gpqafull_*.jsonl).

x, --x uplift: the model's own CoT accuracy (8 samples, no requirement) - its own no-CoT accuracy (24 samples, 'ANSWER:' prefill); y from the seed-0 run (one
sample per type). --x constrained: the model's accuracy on its CoT-control rows (the same rows as y, seed-0 and seed-1 runs, two samples per type) - base's
no-CoT accuracy (48 samples, gpqafull_base_nocot48.jsonl). y, per item, mean over the types applied to it (F0 wording, user turn): strict = strict pass with no instruction mention, all 21 manytypes2 types; partial = the programmatic grader's partial score, the 15
programmatic types only (the 6 judged types have no partial score; mentions not counted). Lines: mean y in fifths of x (95% normal intervals); r: Pearson
correlation over items with a 95% item bootstrap.

    python uplift_scatter.py [--x constrained]
"""
import argparse
import collections
import json

import matplotlib.pyplot as plt
import numpy as np

from difficulty_compliance import RESULTS, SURFACE, TEXT, TEXT2, TYPES, style, rows_for
from gpqa_scatter import MODELS

FIGS = RESULTS.parent.parent.parent / "figures"
XMODES = {"uplift": ("gpqa_uplift_compliance.png", "the model's own CoT accuracy (8 samples) - no-CoT accuracy (24 samples)"),
          "constrained": ("gpqa_constrained_uplift_compliance.png", "the model's accuracy under the CoT-control requirements (2 samples per type) - base's no-CoT accuracy (48 samples)")}


def per_item(key, x_mode):
    strict, partial, correct = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for run in [f"gpqafull-{key}"] + ([f"gpqafull-{key}-seed1"] if x_mode == "constrained" else []):
        for r in rows_for(run, "gpqa"):
            strict[r["item"]].append(r["ok"])
        for r in map(json.loads, open(RESULTS / f"transfer/{run}/gens.jsonl")):
            if r["type"] in TYPES:
                correct[r["item_idx"]].append(r["correct"])
                if r["partial"] is not None:
                    partial[r["item_idx"]].append(r["partial"])
    if x_mode == "uplift":
        acc = {d["item_idx"]: d for d in map(json.loads, open(RESULTS / f"difficulty/gpqafull_{key}.jsonl"))}
        x = {i: d["cot_correct"] / d["cot_samples"] - d["nocot_correct"] / d["samples"] for i, d in acc.items()}
    else:
        base = {d["item_idx"]: d["nocot_correct"] / d["samples"] for d in map(json.loads, open(RESULTS / "difficulty/gpqafull_base_nocot48.jsonl"))}
        x = {i: np.mean(c) - base[i] for i, c in correct.items()}
    return {name: (np.array([x[i] for i in sorted(d)]), np.array([np.mean(d[i]) for i in sorted(d)])) for name, d in [("strict", strict), ("partial", partial)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", default="uplift", choices=list(XMODES))
    args = ap.parse_args()
    out, xlabel = XMODES[args.x]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=SURFACE, layout="constrained", sharex=True)
    for label, key, color in MODELS:
        data = per_item(key, args.x)
        for ax, name in zip(axes, ["strict", "partial"]):
            x, y = data[name]
            idx = rng.integers(0, len(x), (2000, len(x)))
            r_ci = np.percentile([np.corrcoef(x[k], y[k])[0, 1] for k in idx], [2.5, 97.5])
            ax.scatter(x + rng.uniform(-0.012, 0.012, len(x)), 100 * y, s=10, color=color, alpha=0.3, linewidths=0, zorder=3)
            edges = np.quantile(x, np.linspace(0, 1, 6))
            b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 4)
            mx = [x[b == k].mean() for k in range(5) if (b == k).any()]
            my = [100 * y[b == k].mean() for k in range(5) if (b == k).any()]
            se = [196 * y[b == k].std() / np.sqrt((b == k).sum()) for k in range(5) if (b == k).any()]
            ax.errorbar(mx, my, se, color=color, lw=2, marker="o", ms=5, capsize=3, zorder=4,
                        label=f"{label}: r = {np.corrcoef(x, y)[0, 1]:+.2f} [{r_ci[0]:+.2f}, {r_ci[1]:+.2f}], mean {100 * y.mean():.0f}%")
    style(axes[0], "Strict compliance (all 21 types, no instruction mention)", xlabel, "item's strict pass rate over types (%)")
    style(axes[1], "Partial compliance (15 programmatic types)", xlabel, "item's mean partial score over types (%)")
    for ax in axes:
        ax.set_ylim(0, 100)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=9, labelcolor=TEXT)
    fig.supxlabel(f"GPQA, 360 items, {'two samples' if args.x == 'constrained' else 'one sample'} per item and type (F0 wording, user turn). Dots: items (jittered). Lines: mean in fifths of x with 95% intervals. "
                  "r: Pearson over items, 95% item bootstrap.", fontsize=8.5, color=TEXT2)
    fig.savefig(FIGS / out, dpi=130, facecolor=SURFACE)
    print(FIGS / out)


if __name__ == "__main__":
    main()
