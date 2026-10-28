"""Dark summary figure for this run. Run with cot-control/rl/.venv/bin/python; writes PNG and SVG beside the results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / "summary.json").read_text())
rows = [json.loads(line) for line in (HERE / "rollouts.jsonl").read_text().splitlines()]
judged = [json.loads(line) for line in (HERE / "judge-gemini-0-10.jsonl").read_text().splitlines()]
assert summary["completed"] == summary["expected"] == summary["judged"] == len(rows) == len(judged)
COLORS = {"hack": "#f3a45d", "clean": "#67b6ed", "cheated": "#e87d94"}
plt.rcParams.update({"figure.facecolor": "#151b23", "axes.facecolor": "#151b23", "savefig.facecolor": "#151b23", "text.color": "#e6edf3", "axes.labelcolor": "#e6edf3",
                     "xtick.color": "#b9c4d0", "ytick.color": "#b9c4d0", "axes.edgecolor": "#506070", "axes.grid": True, "grid.color": "#293340", "axes.axisbelow": True,
                     "font.size": 11, "axes.titleweight": "bold", "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

ax = axes[0, 0]
tops = []
for i, condition in enumerate(("hack", "clean")):
    for j, side in enumerate(("even", "odd", "both")):
        cell = summary["cells"][f"{condition}/{side}"]
        rate = 100 * cell["rate"]
        lo, hi = [100 * p for p in cell["wilson95"]]
        x = j + (i - 0.5) * 0.35
        ax.bar(x, rate, 0.32, color=COLORS[condition], label={"hack": "Conflicting grader", "clean": "Matching grader"}[condition] if j == 0 else None)
        ax.errorbar(x, rate, [[rate - lo], [hi - rate]], color="#d4dee7", capsize=4)
        ax.annotate(f"{cell['cheat']}/{cell['valid']}", (x, hi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10)
        tops.append(hi)
ax.set_xticks(range(3), ["Even request", "Odd request", "Both pooled"])
ax.set_ylim(0, max(tops + [1]) * 1.4)
ax.set_ylabel("Opposite-parity answers (% valid)")
ax.set_title("Answer parity · 95% Wilson intervals", loc="left")
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.grid(axis="x", visible=False)

ax = axes[0, 1]
for i, condition in enumerate(("hack", "clean")):
    ratings = summary["ratings"][f"{condition}/all"]
    counts = np.array(ratings["histogram"]) / ratings["rated"] * 100
    ax.bar(np.arange(11) + (i - 0.5) * 0.4, counts, 0.38, color=COLORS[condition],
           label=f"{'Conflicting' if condition == 'hack' else 'Matching'} grader · {ratings['rated']}/{ratings['n']} rated")
ax.set_xticks(range(11))
ax.set_xlabel("Meta-reasoning rating (0–10)")
ax.set_ylabel("% of rated CoTs")
ax.set_title("Gemini 3.8 Flash · CoT-only rating distribution", loc="left")
ax.set_ylim(0, max(patch.get_height() for patch in ax.patches) * 1.35)
ax.legend(frameon=False, fontsize=10)
ax.grid(axis="x", visible=False)

ax = axes[1, 0]
groups = [("hack", "all", "Conflicting\nall"), ("hack", "cheated", "Conflicting\ncheated"), ("hack", "noncheating", "Conflicting\nnon-cheating"), ("clean", "all", "Matching\nall")]
for x, (condition, outcome, _) in enumerate(groups):
    selected = [r for r in judged if r["prompts"] == condition and (outcome == "all" or r["cheat"] is (outcome == "cheated"))]
    values = np.array([r["meta"] for r in selected if r["meta"] is not None])
    if not len(values):
        ax.text(x, 0.2, "0/0 rated", ha="center")
        continue
    mean = values.mean()
    ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
    ax.bar(x, mean, 0.65, color=COLORS["cheated" if outcome == "cheated" else condition])
    if len(values) > 1:
        ax.errorbar(x, mean, ci, color="#d4dee7", capsize=5)
    label = f"{mean:.2f}\n{len(values)}/{len(selected)} rated" + ("\nCI unavailable" if len(values) == 1 else "")
    ax.text(x, mean + ci + 0.17, label, ha="center", fontsize=10)
ax.set_xticks(range(4), [g[2] for g in groups])
ax.set_ylim(0, 12.2)
ax.set_yticks(range(0, 11, 2))
ax.set_ylabel("Mean meta-reasoning rating (0–10)")
ax.set_title("Ratings by outcome · 95% normal CIs", loc="left")
ax.grid(axis="x", visible=False)

ax = axes[1, 1]
for condition, outcome, label in (("hack", None, "Conflicting grader"), ("hack", True, "Conflicting · cheated"), ("clean", None, "Matching grader")):
    cell = [r for r in rows if r["prompts"] == condition]
    selected = [r for r in cell if outcome is None or r["cheat"] is outcome]
    values = np.sort([r["n_output_tokens"] for r in selected])
    if not len(values):
        continue
    ax.step(values, np.arange(1, len(values) + 1) / len(values) * 100, where="post", lw=2, marker="o" if len(values) == 1 else None, color=COLORS["cheated" if outcome else condition],
            label=f"{label}: median {np.median(values):,.0f}\n{len(selected)}/{len(cell)} samples")
ax.set_xscale("log")
ax.set_xlabel("Output tokens (reasoning + final answer; log scale)")
ax.set_ylabel("Cumulative % of samples")
ax.set_ylim(0, 104)
ax.set_title("Output lengths", loc="left")
ax.legend(frameon=False, fontsize=10, loc="lower right")

truncated = sum(r["truncated"] for r in rows)
valid = sum(r["cheat"] is not None for r in rows)
fig.suptitle("Base gpt-oss-120b · parity grader-following", fontsize=22, x=0.055, ha="left", y=0.985)
fig.text(0.055, 0.944, "Groq via OpenRouter · medium effort · 256 samples per request parity × grader condition · temperature 1 · top-p 0.95", fontsize=12, color="#b9c4d0")
fig.text(0.055, 0.025, f"CoT-only monitor: hack_judge.RATING_PROMPT; Gemini 3.8 Flash / Google AI Studio, low effort, temperature 0. Request + grader context; no final answer.\n"
                      f"Rating: 0 = only the request; 10 = almost entirely meta-reasoning. Valid answers {valid}/{len(rows)}; token-cap truncations {truncated}/{len(rows)}. 2026-10-27.", fontsize=10, color="#b9c4d0")
fig.tight_layout(rect=(0.025, 0.075, 0.99, 0.925), h_pad=3.0, w_pad=3.0)
for suffix in ("png", "svg"):
    path = HERE / f"parity_summary.{suffix}"
    fig.savefig(path, dpi=160)
    print(path)
