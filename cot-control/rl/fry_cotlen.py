"""Fry measure: how much a CoT-control request changes CoT length, per model, from the transfer eval's generations.

For every item of a transfer run (results/transfer/<run>/gens.jsonl) the baseline arm (type 'baseline', no requirement) and each
requirement type were sampled on the same question, so lengths pair by (dataset, item_idx). Per type: median output tokens with the
requirement, the baseline median on the same items, and the median per-item log2(tokens with requirement / tokens without). Per model:
the median of the per-type ratios, the share of types whose median ratio is below 0.8 or above 1.25, and the no-requirement median
against the reference model's. A model that shortens or lengthens its CoT whenever any requirement is present has changed how it
reasons under instruction, beyond the trained constraint; the reference model's own ratios are the comparison.

    python fry_cotlen.py                                   # default runs, F0 wording, user placement
    python fry_cotlen.py --runs gptoss-base gptoss-grpo3s100 --figure ../../../figures/fry_cotlen_ratio.png

Writes results/fry/<--out>.md and .json (default cotlen), and the figure when --figure is given.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
RESULTS = Path(__file__).resolve().parent.parent / "results"
DEFAULT_RUNS = ["gptoss-base", "gptoss-grpo3s100", "mt1b-s150", "gptoss-len2s60", "gptoss-st1-ignore2", "gptoss-st1-words", "gptoss-st1-third", "gptoss-st1-qonly"]


def load(run, frame, placement):
    rows = [json.loads(l) for l in open(RESULTS / "transfer" / run / "gens.jsonl")]
    base = {(r["dataset"], r["item_idx"]): r["n_output_tokens"] for r in rows if r["type"] == "baseline"}
    by_type = {}
    for r in rows:
        if r["type"] == "baseline" or r["frame"] != frame or r["placement"] != placement:
            continue
        by_type.setdefault(r["type"], []).append((base[(r["dataset"], r["item_idx"])], r["n_output_tokens"]))
    return base, by_type


def summarize(run, frame, placement):
    base, by_type = load(run, frame, placement)
    types = {}
    for t, pairs in sorted(by_type.items()):
        logs = [math.log2(w / wo) for wo, w in pairs]
        types[t] = dict(n=len(pairs), median_with=statistics.median(w for _, w in pairs), median_without=statistics.median(wo for wo, _ in pairs),
                        median_log2_ratio=statistics.median(logs), share_shorter=sum(x < 0 for x in logs) / len(logs))
    ratios = [2 ** v["median_log2_ratio"] for v in types.values()]
    return dict(run=run, baseline_median=statistics.median(base.values()), n_items=len(base), types=types,
                median_type_ratio=statistics.median(ratios), types_below_0_8=sum(r < 0.8 for r in ratios), types_above_1_25=sum(r > 1.25 for r in ratios), n_types=len(ratios))


def table(summaries, ref):
    lines = ["| run | no-requirement median tokens (vs %s) | median over types of (tokens with requirement / without), paired | types with ratio below 0.8 | types above 1.25 | types |" % ref,
             "|---|---|---|---|---|---|"]
    ref_base = next(s["baseline_median"] for s in summaries if s["run"] == ref)
    for s in summaries:
        lines.append(f"| {s['run']} | {s['baseline_median']:.0f} ({s['baseline_median'] / ref_base:.2f}x) | {s['median_type_ratio']:.2f} | {s['types_below_0_8']}/{s['n_types']} | {s['types_above_1_25']}/{s['n_types']} | {s['n_types']} |")
    common = sorted(set.intersection(*(set(s["types"]) for s in summaries)))
    lines += ["", "Per type, paired median ratio (tokens with the requirement / tokens without, same items):", "",
              "| type | " + " | ".join(s["run"] for s in summaries) + " |", "|---|" + "---|" * len(summaries)]
    for t in common:
        lines.append(f"| {t} | " + " | ".join(f"{2 ** s['types'][t]['median_log2_ratio']:.2f}" for s in summaries) + " |")
    return "\n".join(lines)


def figure(summaries, path):
    common = sorted(set.intersection(*(set(s["types"]) for s in summaries)))
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 5.5))
    w = 0.8 / len(summaries)
    for i, s in enumerate(summaries):
        ax.bar([k + i * w for k in range(len(common))], [s["types"][t]["median_log2_ratio"] for t in common], w, label=f"{s['run']} (no-requirement median {s['baseline_median']:.0f} tok)")
    ax.axhline(0, color="white", lw=0.8)
    ax.set_xticks([k + 0.4 - w / 2 for k in range(len(common))])
    ax.set_xticklabels(common, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("median per-item log2(output tokens with requirement / without)")
    ax.set_title("CoT length change under a CoT-control request, by type (transfer eval, F0 wording, user placement)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--ref", default="gptoss-base")
    ap.add_argument("--frame", default="F0")
    ap.add_argument("--placement", default="user")
    ap.add_argument("--figure", default=None)
    ap.add_argument("--out", default="cotlen", help="results/fry/<out>.md and .json")
    ap.add_argument("--figure-runs", nargs="+", default=["gptoss-base", "gptoss-grpo3s100", "mt1b-s150", "gptoss-st1-ignore2"])
    args = ap.parse_args()
    summaries = [summarize(r, args.frame, args.placement) for r in args.runs]
    md = table(summaries, args.ref)
    print(md)
    out = RESULTS / "fry"
    (out / f"{args.out}.md").write_text(md + "\n")
    (out / f"{args.out}.json").write_text(json.dumps(dict(frame=args.frame, placement=args.placement, runs=summaries), indent=1))
    if args.figure:
        figure([s for s in summaries if s["run"] in args.figure_runs], args.figure)
