"""Per-item CoT-control compliance against the model's own CoT uplift on the 360 full-GPQA items (results/transfer/gpqafull-<model>, results/difficulty/gpqafull_<model>.jsonl).

x: CoT accuracy (8 samples) - no-CoT accuracy (24 samples, 'ANSWER:' prefill), the model's own. y, per item, mean over the types applied to it (one sample each,
F0 wording, user turn): strict = strict pass with no instruction mention, all 21 manytypes2 types; partial = the programmatic grader's partial score, the 15
programmatic types only (the 6 judged types have no partial score; mentions not counted). Lines: mean y in fifths of x (95% normal intervals); r: Pearson
correlation over items with a 95% item bootstrap.

    python uplift_scatter.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from difficulty_compliance import RESULTS, SURFACE, TEXT, TEXT2, TYPES, style, rows_for
from gpqa_scatter import MODELS

OUT = RESULTS.parent.parent.parent / "figures/gpqa_uplift_compliance.png"


def per_item(key):
    acc = {d["item_idx"]: d for d in map(json.loads, open(RESULTS / f"difficulty/gpqafull_{key}.jsonl"))}
    up = {i: d["cot_correct"] / d["cot_samples"] - d["nocot_correct"] / d["samples"] for i, d in acc.items()}
    gens = [r for r in map(json.loads, open(RESULTS / f"transfer/gpqafull-{key}/gens.jsonl")) if r["type"] in TYPES]
    strict, partial = {}, {}
    for r in rows_for(f"gpqafull-{key}", "gpqa"):
        strict.setdefault(r["item"], []).append(r["ok"])
    for r in gens:
        if r["partial"] is not None:
            partial.setdefault(r["item_idx"], []).append(r["partial"])
    return {name: (np.array([up[i] for i in sorted(d)]), np.array([np.mean(d[i]) for i in sorted(d)])) for name, d in [("strict", strict), ("partial", partial)]}


def main():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=SURFACE, layout="constrained", sharex=True)
    for label, key, color in MODELS:
        data = per_item(key)
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
    style(axes[0], "Strict compliance (all 21 types, no instruction mention)", "the model's own CoT accuracy (8 samples) - no-CoT accuracy (24 samples)", "item's strict pass rate over types (%)")
    style(axes[1], "Partial compliance (15 programmatic types)", "the model's own CoT accuracy (8 samples) - no-CoT accuracy (24 samples)", "item's mean partial score over types (%)")
    for ax in axes:
        ax.set_ylim(0, 100)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=9, labelcolor=TEXT)
    fig.supxlabel("GPQA, 360 items, one sample per item and type (F0 wording, user turn). Dots: items (jittered). Lines: mean in fifths of x with 95% intervals. "
                  "r: Pearson over items, 95% item bootstrap.", fontsize=8.5, color=TEXT2)
    fig.savefig(OUT, dpi=130, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
