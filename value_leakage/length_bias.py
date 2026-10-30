"""Is a model's donation-bet bias explained by how long it reasons? (Value Leakage appendix E.6 asks this across models;
here it is the confound check for base vs organism.) Rows from the no-line median-threshold cells of the given variants are
binned by reasoning length (quintiles of the pooled word counts, so bins are shared), the bias is computed per model per
bin, and each model's bias is re-weighted to the other model's length distribution (direct standardization; bootstrap CI
by resampling rows within model and direction).

    python length_bias.py --name tinker --tags gptoss-base-tinker grpo3-s100 --names "base gpt-oss-120b (Tinker)" "grpo3 step 100" --variants main accurate f5_naked
    python length_bias.py --name all17 --tags gptoss-base grpo3-s100 --names "base gpt-oss-120b (DeepInfra, n=64)" "grpo3 step 100" --variants all

Writes results/length_bias/<name>.json, <name>.png (copied to mats12_sprint/figures/value_leakage/length_bias.<name>.png).
"""
import argparse
import json
import shutil

import matplotlib
import numpy as np

from bet import DIRECTIONS, good_side
from common import RESULTS, read
from oversight import COLORS, words
from prompts import VARIANTS
from report import C, FIGS, clean

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BINS = 5
DRAWS = 1000


def load(tag: str, variants: list[str]) -> list[dict]:
    rows = [r for v in variants for p in (RESULTS / tag / "bet" / "samples").glob(f"*.{v}.None.None.p50.*.jsonl") for r in read(p)]
    return [{"w": words(r), "d": r["condition"], "g": good_side(r), "q": r["question"], "v": r["variant"]} for r in rows if r.get("estimate") is not None]


def bias_of(rows: list[dict]) -> float:
    ps = [np.mean([r["g"] for r in rows if r["d"] == d]) if any(r["d"] == d for r in rows) else np.nan for d in DIRECTIONS]
    return float(ps[0] + ps[1] - 1)


def bin_index(w: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, w, side="right") - 1, 0, BINS - 1)


def analyse(rows_by_tag: dict, rng) -> dict:
    allw = np.concatenate([[r["w"] for r in rows] for rows in rows_by_tag.values()])
    edges = np.quantile(allw, np.linspace(0, 1, BINS + 1))
    out = {"edges": edges.tolist(), "tags": {}}
    tags = list(rows_by_tag)
    for t in tags:
        rows = rows_by_tag[t]
        w = np.array([r["w"] for r in rows])
        b = bin_index(w, edges)
        out["tags"][t] = {"n": len(rows), "median_words": float(np.median(w)), "iqr_words": [float(np.percentile(w, 25)), float(np.percentile(w, 75))],
                          "bias": bias_of(rows), "bin_share": [float(np.mean(b == i)) for i in range(BINS)],
                          "bin_bias": [bias_of([r for r, bi in zip(rows, b) if bi == i]) for i in range(BINS)],
                          "bin_n": [[int(sum(1 for r, bi in zip(rows, b) if bi == i and r["d"] == d)) for d in DIRECTIONS] for i in range(BINS)],
                          "bin_median_words": [float(np.median(w[b == i])) if (b == i).any() else float("nan") for i in range(BINS)]}
    for t in tags:   # within-question check: tertiles of each question's own lengths (per model), bias per tertile averaged over questions
        rows = rows_by_tag[t]
        qs = sorted({r["q"] for r in rows})
        tert = np.zeros((len(qs), 3))
        for qi, q in enumerate(qs):
            qr = [r for r in rows if r["q"] == q]
            e = np.quantile([r["w"] for r in qr], [0, 1 / 3, 2 / 3, 1])
            b = np.clip(np.searchsorted(e, [r["w"] for r in qr], side="right") - 1, 0, 2)
            tert[qi] = [bias_of([r for r, bi in zip(qr, b) if bi == i]) for i in range(3)]
        out["tags"][t]["tertile_bias_within_question"] = np.nanmean(tert, 0).tolist()
        out["tags"][t]["tertile_bias_per_question"] = {q: tert[qi].tolist() for qi, q in enumerate(qs)}
    for t in tags:   # standardize each model's bin biases to the other model's length distribution, bootstrap over rows within direction
        other = [u for u in tags if u != t][0]
        wts = np.array(out["tags"][other]["bin_share"])
        rows = rows_by_tag[t]
        by_d = {d: [r for r in rows if r["d"] == d] for d in DIRECTIONS}

        def std_bias(sample_rows):
            w = np.array([r["w"] for r in sample_rows])
            b = bin_index(w, edges)
            bb = np.array([bias_of([r for r, bi in zip(sample_rows, b) if bi == i]) for i in range(BINS)])
            return float(np.nansum(wts * bb) / np.sum(wts[~np.isnan(bb)]))
        point = std_bias(rows)
        draws = [std_bias([by_d[d][i] for d in DIRECTIONS for i in rng.integers(0, len(by_d[d]), len(by_d[d]))]) for _ in range(DRAWS)]
        out["tags"][t]["bias_at_other_length"] = {"other": other, "bias": point, "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
        draws0 = [bias_of([by_d[d][i] for d in DIRECTIONS for i in rng.integers(0, len(by_d[d]), len(by_d[d]))]) for _ in range(DRAWS)]
        out["tags"][t]["bias_ci95"] = [float(np.percentile(draws0, 2.5)), float(np.percentile(draws0, 97.5))]
    return out


def figure(res: dict, names: dict, title: str, out) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for j, (t, s) in enumerate(res["tags"].items()):
        x = np.array(s["bin_median_words"])
        y = np.array(s["bin_bias"])
        n = np.array(s["bin_n"])
        se = np.sqrt(sum((0.5 * 0.5) / np.maximum(n[:, k], 1) for k in range(2)))   # conservative binomial SE on p_below + p_above
        ax.errorbar(x, y, yerr=1.96 * se, fmt="o-", color=COLORS[j], ms=6, lw=2, capsize=2, markeredgecolor=C["surface"], markeredgewidth=1.5,
                    label=f"{names[t]}: bias {s['bias']:+.3f} [{s['bias_ci95'][0]:+.3f}, {s['bias_ci95'][1]:+.3f}]; at the other model's lengths {s['bias_at_other_length']['bias']:+.3f} [{s['bias_at_other_length']['ci95'][0]:+.3f}, {s['bias_at_other_length']['ci95'][1]:+.3f}]")
        ax.axhline(s["bias"], color=COLORS[j], lw=1, ls="--", alpha=0.6)
    ax.axhline(0, color=C["axis"], lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("reasoning length, words (shared quintile bins; point at the bin median)")
    ax.set_ylabel("bias within the bin (pooled over questions)")
    ax.set_title("bias by reasoning length; dashed = the model's overall bias", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.14))
    clean(ax, "y")
    ax = axes[1]
    edges = res["edges"]
    for j, (t, s) in enumerate(res["tags"].items()):
        ax.bar(np.arange(BINS) + (j - 0.5) * 0.4, s["bin_share"], width=0.38, color=COLORS[j], label=f"{names[t]}: median {s['median_words']:.0f} words")
    ax.set_xticks(range(BINS), [f"{edges[i]:.0f}-{edges[i + 1]:.0f}" for i in range(BINS)], fontsize=7.5)
    ax.set_xlabel("length bin, words")
    ax.set_ylabel("share of the model's rows")
    ax.set_title("where each model's traces fall", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    clean(ax, "y")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", nargs=2, required=True)
    ap.add_argument("--names", nargs=2, required=True)
    ap.add_argument("--variants", nargs="+", default=["main", "accurate", "f5_naked"])
    args = ap.parse_args()
    variants = list(VARIANTS) if args.variants == ["all"] else args.variants
    rows = {t: load(t, variants) for t in args.tags}
    res = analyse(rows, np.random.default_rng(0))
    res["variants"] = variants
    names = dict(zip(args.tags, args.names))
    for t, s in res["tags"].items():
        print(f"{names[t]}: {s['n']} rows, median {s['median_words']:.0f} words (IQR {s['iqr_words'][0]:.0f}-{s['iqr_words'][1]:.0f}); bias {s['bias']:+.3f} [{s['bias_ci95'][0]:+.3f}, {s['bias_ci95'][1]:+.3f}]; "
              f"at {names[s['bias_at_other_length']['other']]}'s length distribution {s['bias_at_other_length']['bias']:+.3f} [{s['bias_at_other_length']['ci95'][0]:+.3f}, {s['bias_at_other_length']['ci95'][1]:+.3f}]")
        print("   bin bias:", " ".join(f"{b:+.3f}" for b in s["bin_bias"]), "| bin share:", " ".join(f"{x:.2f}" for x in s["bin_share"]), "| rows below/above per bin:", s["bin_n"])
        print("   within-question tertiles (short, mid, long), mean over questions:", " ".join(f"{b:+.3f}" for b in s["tertile_bias_within_question"]),
              "| questions with long < short:", sum(v[2] < v[0] for v in s["tertile_bias_per_question"].values()), "/", len(s["tertile_bias_per_question"]))
    out = RESULTS / "length_bias"
    out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=1))
    figure(res, names, f"Donation bet: is the bias a function of reasoning length? {' + '.join(variants) if len(variants) <= 3 else f'{len(variants)} variants'}, median threshold, no oversight line", out / f"{args.name}.png")
    shutil.copy(out / f"{args.name}.png", FIGS / f"length_bias.{args.name}.png")
    print(out / f"{args.name}.png")


if __name__ == "__main__":
    main()
