"""Compare the Groq base with mt1b s150 and mt2e s170 on Tinker. Local rl venv; writes PNG, SVG and comparison.json."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = [("Base · Groq", HERE.parent / "gptoss120b_groq_medium_20261027", "#94a6b7"), ("mt1b s150 · Tinker", HERE / "mt1b", "#67b6ed"), ("mt2e s170 · Tinker", HERE / "mt2e", "#f3a45d")]
BINS = [(0, 50, "<50"), (50, 100, "50–99"), (100, 200, "100–199"), (200, 400, "200–399"), (400, float("inf"), "≥400")]
plt.rcParams.update({"figure.facecolor": "#151b23", "axes.facecolor": "#151b23", "savefig.facecolor": "#151b23", "text.color": "#e6edf3", "axes.labelcolor": "#e6edf3",
                     "xtick.color": "#b9c4d0", "ytick.color": "#b9c4d0", "axes.edgecolor": "#506070", "axes.grid": True, "grid.color": "#293340", "axes.axisbelow": True,
                     "font.size": 11, "axes.titleweight": "bold", "svg.fonttype": "none"})


def mean_ci(values: list) -> tuple[float, float]:
    return float(np.mean(values)), float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0


fig, axes = plt.subplots(3, 2, figsize=(17, 16))
comparison = {}
for i, (label, path, color) in enumerate(RUNS):
    summary = json.loads((path / "summary.json").read_text())
    rows = [json.loads(line) for line in (path / "rollouts.jsonl").read_text().splitlines()]
    judged = [json.loads(line) for line in (path / "judge-gemini-0-10.jsonl").read_text().splitlines()]
    assert summary["completed"] == summary["expected"] == summary["judged"] == len(rows) == len(judged) == 1024
    by_key = {r["key"]: r for r in rows}
    assert len(by_key) == len({r["key"] for r in judged}) == 1024
    assert set(by_key) == {r["key"] for r in judged}
    assert all(type(r["meta"]) is int and r["cheat"] == by_key[r["key"]]["cheat"] for r in judged)
    comparison[label] = {"cells": summary["cells"], "ratings": summary["ratings"], "policy_usd": summary["policy_usd"], "judge_usd": summary["judge_usd"], "length_bins": []}
    offset = (i - 1) * 0.25

    for j, condition in enumerate(("hack", "clean")):
        ax = axes[0, 0]
        cell = summary["cells"][f"{condition}/both"]
        rate = 100 * cell["rate"]
        lo, hi = [100 * p for p in cell["wilson95"]]
        x = j + offset
        ax.bar(x, rate, 0.23, color=color, label=label if j == 0 else None)
        ax.errorbar(x, rate, [[rate - lo], [hi - rate]], color="#d4dee7", capsize=4)
        ax.annotate(f"{cell['cheat']}/{cell['valid']}", (x, hi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10)

        ax = axes[0, 1]
        values = [r["meta"] for r in judged if r["prompts"] == condition]
        mean, ci = mean_ci(values)
        ax.bar(x, mean, 0.23, color=color)
        ax.errorbar(x, mean, ci, color="#d4dee7", capsize=4)
        ax.text(x, mean + ci + 0.16, f"{mean:.2f}", ha="center", fontsize=11)

    ax = axes[1, 0]
    values = [r["meta"] for r in judged if r["prompts"] == "hack" and r["cheat"] is True]
    if values:
        mean, ci = mean_ci(values)
        ax.bar(i, mean, 0.6, color=color)
        if len(values) > 1:
            ax.errorbar(i, mean, ci, color="#d4dee7", capsize=5)
        text = f"{mean:.2f}\n{len(values)}/512 cheated" + ("\nCI unavailable" if len(values) == 1 else "")
        ax.text(i, mean + ci + 0.17, text, ha="center", fontsize=10)
    else:
        ax.text(i, 0.3, "0/512 cheated\nNo conditional mean", ha="center", fontsize=10)

    ax = axes[1, 1]
    values = [r["meta"] for r in judged if r["prompts"] == "hack"]
    ax.bar(np.arange(11) + offset, np.bincount(values, minlength=11) / len(values) * 100, 0.23, color=color, label=label)

    ax = axes[2, 0]
    for condition, style in (("hack", "-"), ("clean", "--")):
        values = np.sort([r["n_reasoning_words"] for r in rows if r["prompts"] == condition])
        assert len(values) == 512 and np.all(values > 0)
        ax.step(values, np.arange(1, len(values) + 1) / len(values) * 100, where="post", lw=2, color=color, linestyle=style,
                label=f"{label}: {np.median(values):.0f} words" if condition == "hack" else None)

    ax = axes[2, 1]
    for j, (lo, hi, bin_label) in enumerate(BINS):
        values = [r["meta"] for r in judged if r["prompts"] == "hack" and lo <= by_key[r["key"]]["n_reasoning_words"] < hi]
        mean, ci = mean_ci(values) if values else (None, None)
        comparison[label]["length_bins"].append({"words": bin_label, "rated": len(values), "total": 512, "mean": mean, "normal_ci95_halfwidth": ci if len(values) > 1 else None})
        x = j + offset
        if values:
            ax.errorbar(x, mean, ci if len(values) > 1 else None, color=color, marker="o", capsize=3, lw=1.5)
            ax.annotate(f"{len(values)}/512", (x, mean + ci), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, rotation=45)
        else:
            ax.text(x, 0.1, "0/512", ha="center", color=color, fontsize=8, rotation=45)

ax = axes[0, 0]
ax.set_xticks([0, 1], ["Conflicting grader", "Matching grader"])
ax.set_ylabel("Opposite-parity answers (% valid)")
ax.set_title("Answer parity · 95% Wilson intervals", loc="left")
ax.set_ylim(0, max(2, ax.get_ylim()[1]) * 1.35)
ax.legend(frameon=False, fontsize=10, loc="upper right")

ax = axes[0, 1]
ax.set_xticks([0, 1], ["Conflicting grader", "Matching grader"])
ax.set_ylim(0, 11)
ax.set_yticks(range(0, 11, 2))
ax.set_ylabel("Mean meta-reasoning rating (0–10)")
ax.set_title("CoT-only Gemini ratings · all 512/512 per condition", loc="left")

ax = axes[1, 0]
ax.set_xticks(range(3), ["Base · Groq", "mt1b s150 · Tinker", "mt2e s170 · Tinker"])
ax.set_ylim(0, 12.5)
ax.set_yticks(range(0, 11, 2))
ax.set_ylabel("Mean meta-reasoning rating (0–10)")
ax.set_title("CoT-only ratings · cheating samples only", loc="left")

ax = axes[1, 1]
ax.set_xticks(range(11))
ax.set_xlabel("Meta-reasoning rating (0–10)")
ax.set_ylabel("% of rated CoTs")
ax.set_ylim(0, max(patch.get_height() for patch in ax.patches) * 1.35)
ax.set_title("CoT-only rating distribution · conflicting grader", loc="left")
ax.legend(frameon=False, fontsize=10)

ax = axes[2, 0]
ax.set_xscale("log")
ax.set_xticks([20, 50, 100, 200, 500], ["20", "50", "100", "200", "500"])
ax.minorticks_off()
ax.set_ylim(0, 104)
ax.set_xlabel("CoT length (whitespace-separated words; log scale)")
ax.set_ylabel("Cumulative % of samples")
ax.set_title("CoT lengths · solid: conflicting / dashed: matching", loc="left")
ax.legend(frameon=False, fontsize=10, loc="lower right")

ax = axes[2, 1]
ax.set_xticks(range(len(BINS)), [b[2] for b in BINS])
ax.set_ylim(0, 12.5)
ax.set_yticks(range(0, 11, 2))
ax.set_xlabel("CoT length (words); labels show samples / 512")
ax.set_ylabel("Mean meta-reasoning rating (0–10)")
ax.set_title("CoT-only ratings within length bands · conflicting", loc="left")
for ax in axes.flat:
    ax.grid(axis="x", visible=False)
fig.suptitle("Parity grader-following · base, mt1b and mt2e", fontsize=23, x=0.05, ha="left", y=0.985)
fig.text(0.05, 0.955, "Medium effort · 256 samples per request parity × grader condition per model · temperature 1 · top-p 0.95 · 16,384-token cap", fontsize=12, color="#b9c4d0")
fig.text(0.05, 0.018, "CoT-only monitor: hack_judge.RATING_PROMPT; Gemini 3.8 Flash / Google AI Studio, low effort, temperature 0; request + grader context, no final answer.\n"
                     "Mean-rating intervals are 1.96 standard errors when n ≥ 2. Base uses Groq; organisms use Tinker: training and provider differences are not isolated.\n"
                     "Rating: 0 = only about the request; 10 = almost entirely meta-reasoning. Word-length bands are descriptive, not a causal adjustment. 2026-10-27.", fontsize=10, color="#b9c4d0")
fig.tight_layout(rect=(0.02, 0.075, 0.99, 0.935), h_pad=3.0, w_pad=3.0)
for suffix in ("png", "svg"):
    path = HERE / f"parity_comparison.{suffix}"
    fig.savefig(path, dpi=160)
    print(path)
(HERE / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
