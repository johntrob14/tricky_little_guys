"""Per-item CoT-control compliance against no-CoT accuracy on 360 GPQA items, gpt-oss-120b base and CoT-control organisms.

Items: transfer_eval.py --item-set gpqafull (every GPQA item with keywords, minus manytypes2e's GPQA training questions). Compliance: strict pass with no
instruction mention (nometa), F0 wording, user turn, the 21 manytypes2 types, one sample per item and type (results/transfer/gpqafull-<model>). An item's dot
is the model's overall pass rate plus the item's mean over types of (pass - the type's pass rate). Accuracy: item_difficulty.py, 24 no-CoT samples
('ANSWER:' prefill) and 8 CoT samples per item (results/difficulty/gpqafull_<model>.jsonl). Fits as difficulty_compliance.py: logistic with type
intercepts (L2 0.01), x linear and centered, plus log2 output tokens for the length-controlled fit (curve held at the model's median); bands 95% over item bootstraps.

gpqa_scatter.png: rows = each model's own accuracies / base's accuracies; columns = no-CoT accuracy, no-CoT accuracy length-controlled, CoT uplift
(CoT accuracy - no-CoT accuracy). gpqa_scatter_types.png: one panel per type, logistic fit on the model's own no-CoT accuracy, dots = pass rate in
fifths of it (bins with >= 10 items).

    python gpqa_scatter.py
"""
import argparse
import collections
import json

import matplotlib.pyplot as plt
import numpy as np

from difficulty_compliance import RESULTS, SURFACE, TEXT, TEXT2, TYPES, boot, curves, rows_for, style

FIGS = RESULTS.parent.parent.parent / "figures"
MODELS = [("base", "base", "#3987e5"), ("grpo3 s100", "grpo3s100", "#d95926"), ("manytypes1b s150", "mt1b-s150", "#199e70"), ("manytypes2e s170", "mt2e-s170", "#9085e9")]
TRAINED = {"grpo3": ["word_suppression", "multiple_word_suppression", "repeat_sentences", "lowercase_thinking", "meow_between_words", "alternating_case"],
           "mt1b": ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking", "alternating_case", "questions_only",
                    "no_calculation", "no_backtracking", "no_commitment"],
           "mt2e": ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking", "alternating_case", "questions_only",
                    "no_backtracking", "no_commitment", "decoy_topic"]}
X01, XUP = np.linspace(0, 1, 41), np.linspace(-1, 1, 81)


def load(key):
    acc = {d["item_idx"]: d for d in map(json.loads, open(RESULTS / f"difficulty/gpqafull_{key}.jsonl"))}
    nocot = {i: d["nocot_correct"] / d["samples"] for i, d in acc.items()}
    return rows_for(f"gpqafull-{key}", "gpqa"), nocot, {i: d["cot_correct"] / d["cot_samples"] - nocot[i] for i, d in acc.items()}


def per_item(rows):
    tm = {t: np.mean([r["ok"] for r in rows if r["type"] == t]) for t in {r["type"] for r in rows}}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["item"]].append(r["ok"] - tm[r["type"]])
    return {i: np.mean([r["ok"] for r in rows]) + np.mean(v) for i, v in by.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    data = {label: load(key) for label, key, _ in MODELS}
    _, base_nocot, base_up = data["base"]
    rng, summary = np.random.default_rng(0), {}
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), facecolor=SURFACE, layout="constrained")
    for label, key, color in MODELS:
        rows, nocot, up = data[label]
        y = per_item(rows)
        ref = float(np.median([r["tokens"] for r in rows]))
        for ri, (who, xn, xu) in enumerate([("own", nocot, up), ("base's", base_nocot, base_up)]):
            for ci, (name, x, with_len, grid) in enumerate([("nocot", xn, False, X01), ("nocot_lenctl", xn, True, X01), ("uplift", xu, False, XUP)]):
                c, s = curves(rows, x, with_len, ref, grid)
                (lo, hi), sci = boot(rows, x, with_len, ref, n=args.boot, grid=grid)
                ax = axes[ri, ci]
                if not with_len:
                    items = sorted(y)
                    ax.scatter([x[i] + j for i, j in zip(items, rng.uniform(-0.01, 0.01, len(items)))], [100 * y[i] for i in items], s=9, color=color, alpha=0.3, linewidths=0, zorder=3)
                ax.plot(grid, 100 * c, color=color, lw=2, zorder=4, label=label)
                ax.fill_between(grid, 100 * lo, 100 * hi, color=color, alpha=0.12, linewidth=0, zorder=2)
                summary[f"{who} {name} | {label}"] = dict(slope=round(float(s), 3), slope_ci=[round(float(v), 3) for v in sci], n_items=len(y), n_rows=len(rows), median_tokens=ref)
    for ri, who in enumerate(["the model's own", "base's"]):
        style(axes[ri, 0], f"Compliance vs {who} no-CoT accuracy (dots: items, type-adjusted)", f"{who} no-CoT accuracy (24 samples, ANSWER: prefill; dots jittered)", "strict pass, no mention (%)")
        style(axes[ri, 1], f"Same fit at each model's median output length (length-controlled)", f"{who} no-CoT accuracy", "strict pass, no mention (%)")
        style(axes[ri, 2], f"Compliance vs {who} CoT uplift", f"{who} CoT accuracy (8 samples) - no-CoT accuracy (24 samples)", "strict pass, no mention (%)")
        for ax in axes[ri]:
            ax.set_ylim(0, 100)
    fig.legend(*axes[0, 1].get_legend_handles_labels(), loc="outside upper center", ncol=len(MODELS), frameon=False, fontsize=10.5, labelcolor=TEXT)
    fig.supxlabel("GPQA, 360 items (manytypes2e's GPQA training questions dropped), F0 wording, instruction in the user turn, 21 types, one sample per item and type. "
                  "Fits: logistic, type intercepts (L2 0.01), x linear and centered, plus log2 tokens for the controlled fit; bands 95% over item bootstraps.", fontsize=8.5, color=TEXT2)
    fig.savefig(FIGS / "gpqa_scatter.png", dpi=130, facecolor=SURFACE)

    fig, axes = plt.subplots(3, 7, figsize=(26, 11), facecolor=SURFACE, layout="constrained", sharex=True, sharey=True)
    bins = np.linspace(0, 1, 6)
    for ax, t in zip(axes.flat, TYPES):
        for label, key, color in MODELS:
            rows, nocot, _ = data[label]
            rt = [r for r in rows if r["type"] == t]
            c, _ = curves(rt, nocot, False, 1.0, X01)
            ax.plot(X01, 100 * c, color=color, lw=1.8, zorder=4, label=label)
            xs, oks = np.array([nocot[r["item"]] for r in rt]), np.array([r["ok"] for r in rt])
            k = np.digitize(xs, bins[1:-1])
            pts = [(xs[k == b].mean(), 100 * oks[k == b].mean()) for b in range(5) if (k == b).sum() >= 10]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, color=color, zorder=5, edgecolors=SURFACE, linewidths=1)
        trained = [m for m, ts in TRAINED.items() if t in ts]
        style(ax, f"{t}\n{'trained: ' + ', '.join(trained) if trained else 'held out for all'}", "", "")
        ax.set_ylim(0, 100)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=len(MODELS), frameon=False, fontsize=10.5, labelcolor=TEXT)
    fig.supxlabel("each model's own no-CoT accuracy (24 samples). Lines: logistic fit per type; dots: pass rate in fifths of no-CoT accuracy (bins with >= 10 items). "
                  "y: strict pass, no mention (%). GPQA, 360 items, one sample per item and type.", fontsize=9, color=TEXT2)
    fig.savefig(FIGS / "gpqa_scatter_types.png", dpi=110, facecolor=SURFACE)
    (RESULTS / "difficulty/gpqa_scatter.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(k, v)
    print(FIGS / "gpqa_scatter.png", FIGS / "gpqa_scatter_types.png")


if __name__ == "__main__":
    main()
