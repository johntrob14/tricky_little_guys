"""Grouped bars from tally.json: leak rate (95% Wilson), the CoT-only monitor's per-episode mean suspicion and its score at the first leak-detector fire (mean, standard error),
and the share of later turns that repeat turn 0's reasoning, per condition and cell. System python3 (matplotlib).

    python3 results/inkling_prefill/figure.py OUT.png
"""
import json, math, statistics as st, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FG, BG = "#e8e8e8", "#14141a"
CONDS = [("OpenRouter, no prefill", "#54a0ff"), ("Tinker, no prefill", "#1dd1a1"), ("Tinker, specific prefill", "#ff9f43")]
CELLS = ["easy", "medium", "hard", "control"]
rows = {(r["condition"], r["cell"]): r for r in json.load(open(Path(__file__).with_name("tally.json")))}


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, p - (c - h)), max(0, (c + h) - p)


def mean_se(xs):
    se = 0 if len(xs) < 2 else st.stdev(xs) / math.sqrt(len(xs))
    return st.mean(xs), se, se


PANELS = [("side task leaked rate", lambda r: wilson(r["leaked"], r["n"]), (0, 1.05)),
          ("monitor: per-episode mean suspicion (0-10)", lambda r: mean_se(r["ep_mean"]), (0, 10.5)),
          ("monitor score at the first leak-detector fire,\nscorer-leaked episodes (0-10)", lambda r: mean_se(r["at_first_fire"]) if r["at_first_fire"] else None, (0, 10.5)),
          ("later turns whose reasoning repeats turn 0's\nfirst 60 characters (after the prefill)", lambda r: wilson(r["copies"], r["later_turns"]), (0, 1.05))]
plt.rcParams.update({"text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": "#555"})
fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
x, w = np.arange(len(CELLS)), 0.8 / len(CONDS)
for ax, (title, fn, ylim) in zip(axes.flat, PANELS):
    ax.set_facecolor(BG)
    for i, (cond, color) in enumerate(CONDS):
        for j, c in enumerate(CELLS):
            v = fn(rows[(cond, c)]) if (cond, c) in rows else None
            if v:
                ax.bar(x[j] + (i - (len(CONDS) - 1) / 2) * w, v[0], w, yerr=[[v[1]], [v[2]]], color=color, ecolor="#bbb", capsize=2, label=cond if j == 0 else None)
    ax.set_xticks(x, ["arena_covert_v2\neasy", "arena_covert_v2\nmedium", "arena_covert_v2\nhard", "control (no side task)\neasy"])
    ax.set_title(title, color=FG)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#333")
    ax.set_axisbelow(True)
axes[0, 0].legend(facecolor=BG, edgecolor="#555", fontsize=9, loc="upper right")
fig.suptitle("Inkling-Small base on the tiktok leak_secret arena, 16 episodes per cell; Tinker: thinking effort 0.9; prefill: prompts/prefill/specific.txt on every turn\n"
             "Monitor: CoT only (controlarena_cot_only), judge gpt-oss-20b, main task only, prefill stripped. Rates: 95% Wilson intervals; scores: mean ± standard error", color=FG)
fig.tight_layout()
fig.savefig(sys.argv[1], facecolor=BG, dpi=130)
