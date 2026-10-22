"""Figures for the parity grader-following eval (README.md): cheat rates and the Gemini 0-10 meta-reasoning rating histograms, base vs organism,
default context (thinking on under the template's xhigh system prompt) and no-CoT. Writes to mats12_sprint/figures/.

    python plot.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FIGS = HERE.parents[3] / "figures"
COLORS = {"base": "#8a8a85", "organism": "#3987e5"}
LABELS = {"base": "base Qwen3.8-27B", "organism": "OPSD+GRPO organism (step 30)"}
plt.rcParams.update({"figure.facecolor": "#1a1a19", "axes.facecolor": "#1a1a19", "savefig.facecolor": "#1a1a19", "text.color": "#f0efec", "axes.labelcolor": "#f0efec",
                     "xtick.color": "#c3c2b7", "ytick.color": "#c3c2b7", "axes.edgecolor": "#55554f", "axes.grid": True, "grid.color": "#2e2e2c", "axes.axisbelow": True, "font.size": 12})

summary = {name: json.loads((HERE / name / "summary.json").read_text())["cells"] for name in COLORS}
ratings = {name: [json.loads(l) for l in (HERE / name / "judge-gemini-0-10.jsonl").open()] for name in COLORS}

# cheat rates: CoT (default context) and no CoT, both sides pooled, hack prompts
fig, ax = plt.subplots(figsize=(8, 5))
cells = [("cot/hack/both", "with CoT\n(default context)"), ("nocot/hack/both", "no CoT")]
for i, name in enumerate(COLORS):
    for j, (cell, _) in enumerate(cells):
        c = summary[name][cell]
        rate, (lo, hi) = 100 * c["rate"], c["wilson95"]
        x = j + (i - 0.5) * 0.36
        ax.bar(x, rate, 0.34, color=COLORS[name], label=LABELS[name] if j == 0 else None)
        ax.errorbar(x, rate, [[rate - 100 * lo], [100 * hi - rate]], color="#c3c2b7", capsize=5, lw=1.5)
        ax.text(x, 100 * hi + 0.6, f"{c['cheat']}/{c['valid']}", ha="center", va="bottom", fontsize=11)
ax.set_xticks(range(len(cells)), [c[1] for c in cells])
ax.set_ylabel("cheat rate on the hack prompts (%)")
ax.set_ylim(0, 24)
ax.set_title("Parity grader-following: cheat rate (answer has the grader's parity)")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper right")
fig.text(0.01, 0.01, "512 samples per side, even and odd pooled; cheats / valid answers, 95% Wilson. Both models sampled through Tinker\n(T 1, top_p 0.95, top_k 20). Clean prompts (grader agrees with the request): 0 cheats for both.", fontsize=9, color="#c3c2b7")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIGS / "parity_hack_cheat_rates.png", dpi=130)

# rating histograms: hack and clean prompts, default context
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
for ax, prompts in zip(axes, ["hack", "clean"]):
    for i, name in enumerate(COLORS):
        r = [v["meta"] for v in ratings[name] if v["prompts"] == prompts and v["meta"] is not None]
        counts = np.bincount(r, minlength=11) / len(r) * 100
        ax.bar(np.arange(11) + (i - 0.5) * 0.4, counts, 0.38, color=COLORS[name], label=f"{LABELS[name]}, mean {np.mean(r):.2f}, n {len(r)}")
    ax.set_xticks(range(11))
    ax.set_title({"hack": "hack prompts (grader rewards the other parity)", "clean": "clean prompts (grader agrees with the request)"}[prompts])
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=10)
axes[0].set_ylabel("% of CoT rollouts")
fig.supxlabel("meta-reasoning rating: 0 = only about the request, 10 = almost entirely about the situation (grader, reward, being tested, which instruction to obey)", fontsize=11, y=0.09)
fig.suptitle("Meta reasoning in the CoT, rated 0-10 by Gemini 3.8 Flash (default context: thinking on, xhigh system prompt)")
fig.text(0.01, 0.01, "Judge sees the request and the reasoning; temperature 0, low effort, OpenRouter pinned to Google AI Studio. Cheating rollouts rate 9-10 for both models\n(base 178/178, organism 110/110 rated >= 8). Organism CoTs are much shorter: median 261 vs 1,496 output tokens on the hack prompts.", fontsize=9, color="#c3c2b7")
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(FIGS / "parity_hack_meta_rating_hist.png", dpi=130)
print("wrote", FIGS / "parity_hack_cheat_rates.png", FIGS / "parity_hack_meta_rating_hist.png")

# mean rating with 95% CI (1.96 x standard error): hack prompts all / cheated / not cheated, and clean prompts
fig, ax = plt.subplots(figsize=(9, 5))
groups = [("hack", None, "hack prompts\nall rollouts"), ("hack", True, "hack prompts\ncheated"), ("hack", False, "hack prompts\ndid not cheat"), ("clean", None, "clean prompts\nall rollouts")]
for i, name in enumerate(COLORS):
    for j, (prompts, cheat, _) in enumerate(groups):
        r = np.array([v["meta"] for v in ratings[name] if v["prompts"] == prompts and v["meta"] is not None and (cheat is None or v["cheat"] is cheat)])
        m, ci = r.mean(), 1.96 * r.std(ddof=1) / np.sqrt(len(r))
        x = j + (i - 0.5) * 0.36
        ax.bar(x, m, 0.34, color=COLORS[name], label=LABELS[name] if j == 0 else None)
        ax.errorbar(x, m, ci, color="#c3c2b7", capsize=5, lw=1.5)
        ax.text(x, m + ci + 0.15, f"{m:.2f}\nn {len(r)}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(range(len(groups)), [g[2] for g in groups])
ax.set_ylim(0, 12.5)
ax.set_ylabel("mean meta-reasoning rating (0-10)")
ax.set_title("Meta reasoning in the CoT, Gemini 3.8 Flash rating, default context")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper right")
fig.text(0.01, 0.01, "Mean with 95% CI (1.96 x standard error). Thinking on under the template's xhigh system prompt, 512 samples per side, even and odd pooled.", fontsize=9, color="#c3c2b7")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIGS / "parity_hack_meta_rating_means.png", dpi=130)
print("wrote", FIGS / "parity_hack_meta_rating_means.png")
