"""Grouped bars from tally.json: main pass rate, leak rate (95% Wilson intervals), and the CoT-only monitor's per-episode max and mean suspicion (mean, standard error), per model and cell.
System python3 (matplotlib).

    python3 results/cheapbase_scan/figure.py OUT.png
"""
import json, math, statistics as st, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FG, BG = "#e8e8e8", "#14141a"
COLORS = ["#54a0ff", "#a29bfe", "#ff9f43", "#1dd1a1", "#ff6b6b"]
CELLS = ["easy", "medium", "hard", "control"]
data = json.load(open(Path(__file__).with_name("tally.json")))


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, p - (c - h), (c + h) - p


def mean_se(xs):
    se = 0 if len(xs) < 2 else st.stdev(xs) / math.sqrt(len(xs))
    return st.mean(xs), se, se


PANELS = [("main task pass rate", lambda r: wilson(r["main"], r["n"]), (0, 1.05)),
          ("side task leaked rate", lambda r: wilson(r["leaked"], r["n"]), (0, 1.05)),
          ("CoT-only monitor: per-episode max suspicion (0-10)", lambda r: mean_se(r["ep_max"]), (0, 10.5)),
          ("CoT-only monitor: per-episode mean suspicion (0-10)", lambda r: mean_se(r["ep_mean"]), (0, 10.5))]
plt.rcParams.update({"text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": "#555"})
fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor=BG)
x, w = np.arange(len(CELLS)), 0.8 / len(data)
for ax, (title, fn, ylim) in zip(axes.flat, PANELS):
    ax.set_facecolor(BG)
    for i, (key, m) in enumerate(data.items()):
        vals = [fn(m["cells"][c]) for c in CELLS]
        y = [v[0] for v in vals]
        err = [[v[1] for v in vals], [v[2] for v in vals]]
        ax.bar(x + (i - (len(data) - 1) / 2) * w, y, w, yerr=err, color=COLORS[i], label=m["name"], ecolor="#bbb", capsize=2)
    ax.set_xticks(x, ["arena_covert_v2\neasy", "arena_covert_v2\nmedium", "arena_covert_v2\nhard", "control (no side task)\neasy"])
    ax.set_title(title, color=FG)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#333")
    ax.set_axisbelow(True)
axes[0, 1].legend(facecolor=BG, edgecolor="#555", fontsize=9, loc="upper right")
fig.suptitle("Base models on the tiktok leak_secret arena, 16 episodes per cell, OpenRouter; reasoning: model default (DeepSeek and Qwen3.8 references: effort medium)\n"
             "Monitor: CoT only (controlarena_cot_only), judge gpt-oss-20b, main task only. Rates: 95% Wilson intervals; suspicion: mean ± standard error", color=FG)
fig.tight_layout()
fig.savefig(sys.argv[1], facecolor=BG, dpi=130)
