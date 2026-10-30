"""One figure over every model with a CoT-length fry measurement (results/fry/cotlen*.json): the no-request CoT length as a ratio to the
family's base (the fry reading: unconditional change = fried) and the median over types of the paired with/without ratio (conditional
switching, fine). Bars grouped by family; base is drawn as the reference in each group.

    python fry_cotlen_summary.py --out ../../../figures/fry_cotlen_summary.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
FRY = Path(__file__).resolve().parent.parent / "results" / "fry"
FAMILIES = [("gpt-oss-120b", "cotlen.json", "gptoss-base"), ("Qwen3.8-27B", "cotlen_qwen.json", "qwen-base"), ("DeepSeek V4.1 Flash", "cotlen_deepseek.json", "ds-base")]
LABELS = {"gptoss-base": "base", "gptoss-grpo3s100": "grpo3 s100", "mt1b-s150": "manytypes1b s150", "gptoss-len2s60": "len2 s60", "gptoss-st1-ignore2": "st1-ignore2",
          "gptoss-st1-words": "st1-words", "gptoss-st1-third": "st1-third", "gptoss-st1-qonly": "st1-qonly", "qwen-base": "base", "qwen-opsd1s30": "opsd1 s30",
          "qwen-opsdgrpo": "opsd-grpo1b s10", "qwen-ws1tutor1s2": "ws1tutor1 s2", "ds-base": "base", "ds-rwsft1ep2": "rwsft1 ep2", "ds-obf2gs25": "obf2g s25"}
COLORS = ["#7fc8f8", "#f8c87f", "#b8f87f"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = 0
    ticks, names = [], []
    for (fam, f, base), color in zip(FAMILIES, COLORS):
        runs = json.load(open(FRY / f))["runs"]
        ref = next(r["baseline_median"] for r in runs if r["run"] == base)
        for r in runs:
            uncond, cond = r["baseline_median"] / ref, r["median_type_ratio"]
            axes[0].bar(x, uncond, 0.8, color=color, alpha=1.0 if r["run"] != base else 0.45)
            axes[0].text(x, uncond + 0.03, f"{uncond:.2f}", ha="center", fontsize=8)
            axes[1].bar(x, cond, 0.8, color=color, alpha=1.0 if r["run"] != base else 0.45)
            axes[1].text(x, cond + 0.02, f"{cond:.2f}", ha="center", fontsize=8)
            ticks.append(x); names.append(f"{LABELS.get(r['run'], r['run'])}\n({fam.split()[0]})")
            x += 1
        x += 1
    for ax, title, ylab in [(axes[0], "Unconditional: no-request median CoT tokens / family base", "ratio to base"),
                            (axes[1], "Conditional: median over types of (tokens with requirement / without), paired", "ratio")]:
        ax.axhline(1.0, color="white", lw=0.8, ls="--")
        ax.set_xticks(ticks); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylab)
    axes[0].text(0.01, 0.97, "fry reading: distance from 1.0 here = fried (faded bar = that family's base)", transform=axes[0].transAxes, fontsize=8, va="top")
    axes[1].text(0.01, 0.97, "length switching under a request is fine; compare with the base's own value", transform=axes[1].transAxes, fontsize=8, va="top")
    fig.suptitle("CoT length change with and without a CoT-control request (transfer eval, F0 wording, user placement)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
