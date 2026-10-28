"""Inkling Small parity and CoT-only monitor results. Run with the local rl venv."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = [("User metadata", "user", "#94a6b7"), ("System metadata", "system", "#67b6ed")]
plt.rcParams.update({"figure.facecolor": "#151b23", "axes.facecolor": "#151b23", "savefig.facecolor": "#151b23", "text.color": "#e6edf3", "axes.labelcolor": "#e6edf3",
                     "xtick.color": "#b9c4d0", "ytick.color": "#b9c4d0", "axes.edgecolor": "#506070", "axes.grid": True, "grid.color": "#293340", "axes.axisbelow": True,
                     "font.size": 11, "axes.titleweight": "bold", "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
comparison = {}
for i, (label, variant, color) in enumerate(RUNS):
    path = HERE / variant
    summary = json.loads((path / "summary.json").read_text())
    rows = [json.loads(line) for line in (path / "rollouts.jsonl").read_text().splitlines()]
    judged = [json.loads(line) for line in (path / "judge-gemini-0-10.jsonl").read_text().splitlines()]
    assert summary["completed"] == summary["expected"] == summary["judged"] == len(rows) == len(judged) == 512
    comparison[variant] = {key: summary[key] for key in ("cells", "ratings", "policy_usd", "judge_usd")}
    offset = (i - .5) * .35
    for j, side in enumerate(("even", "odd", "both")):
        ax = axes[0, 0]
        cell = summary["cells"][f"hack/{side}"]
        rate = 100 * cell["rate"]
        lo, hi = [100 * p for p in cell["wilson95"]]
        ax.bar(j + offset, rate, .32, color=color, label=label if j == 0 else None)
        ax.errorbar(j + offset, rate, [[max(0, rate - lo)], [max(0, hi - rate)]], color="#d4dee7", capsize=4)
        ax.annotate(f"{cell['cheat']}/{cell['valid']}", (j + offset, hi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10)
    for j, outcome in enumerate(("all", "cheated", "noncheating")):
        ax = axes[0, 1]
        values = np.array([r["meta"] for r in judged if outcome == "all" or r["cheat"] is (outcome == "cheated")])
        if not len(values):
            ax.text(j + offset, .2, "0/0 rated", ha="center", fontsize=9)
            continue
        mean = values.mean()
        ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        ax.bar(j + offset, mean, .32, color=color)
        if len(values) > 1:
            ax.errorbar(j + offset, mean, ci, color="#d4dee7", capsize=4)
        ax.text(j + offset, mean + ci + .15, f"{mean:.2f}\n{len(values)}/{len(values)} rated", ha="right" if i == 0 else "left", fontsize=9)
    values = [r["meta"] for r in judged]
    axes[1, 0].bar(np.arange(11) + offset, np.bincount(values, minlength=11) / len(values) * 100, .32, color=color, label=label)
    values = np.sort([r["n_reasoning_words"] for r in rows])
    axes[1, 1].step(values, np.arange(1, len(values) + 1) / len(values) * 100, where="post", lw=2, color=color, label=f"{label}: median {np.median(values):.0f} words")

ax = axes[0, 0]
ax.set_xticks([0, 1, 2], ["Request even", "Request odd", "Pooled"])
ax.set_ylabel("Opposite-parity answers (% valid)")
ax.set_title("Answer parity · 95% Wilson intervals", loc="left")
ax.set_ylim(0, max(2, ax.get_ylim()[1]) * 1.3)
ax.legend(frameon=False)
ax = axes[0, 1]
ax.set_xticks([0, 1, 2], ["All samples", "Opposite parity", "Followed parity"])
ax.set_ylim(0, 12.5)
ax.set_yticks(range(0, 11, 2))
ax.set_ylabel("Mean meta-reasoning rating (0–10)")
ax.set_title("CoT-only Gemini ratings · 95% normal CIs", loc="left")
ax = axes[1, 0]
ax.set_xticks(range(11))
ax.set_xlabel("Meta-reasoning rating (0–10)")
ax.set_ylabel("% of rated CoTs")
ax.set_title("CoT-only rating distribution", loc="left")
ax.legend(frameon=False)
ax = axes[1, 1]
ax.set_xscale("log")
ax.set_ylim(0, 104)
ax.set_xlabel("CoT length (whitespace-separated words; log scale)")
ax.set_ylabel("Cumulative % of samples")
ax.set_title("CoT lengths", loc="left")
ax.legend(frameon=False, loc="lower right", fontsize=10)
for ax in axes.flat:
    ax.grid(axis="x", visible=False)
fig.suptitle("Inkling Small · user vs system grader metadata", fontsize=21, x=.055, ha="left", y=.985)
fig.text(.055, .945, "DeepInfra via OpenRouter · medium effort · 256 samples per requested parity × metadata placement · conflicting graders only", fontsize=11, color="#b9c4d0")
fig.text(.055, .025, "CoT-only monitor: hack_judge.RATING_PROMPT; Gemini 3.8 Flash / Google AI Studio, low effort, temperature 0.\n"
                   "Judge request context: flattened number request + metadata, no role labels or final answer. Policy: T 1, top-p 0.95, 16,384-token cap.\n"
                   "System placement changes instruction priority. Mean-rating intervals are 1.96 standard errors when n ≥ 2. 2026-10-27.", fontsize=9, color="#b9c4d0")
fig.tight_layout(rect=(.025, .09, .99, .925), h_pad=3, w_pad=3)
for suffix in ("png", "svg"):
    path = HERE / f"parity_metadata_role.{suffix}"
    fig.savefig(path, dpi=160)
    print(path)
(HERE / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
