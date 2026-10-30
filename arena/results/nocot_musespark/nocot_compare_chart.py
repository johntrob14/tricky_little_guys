"""Figure for nocot_compare.py: reasoning on vs no CoT per model, the three arena_covert_v2 tiers pooled (48 episodes per bar). Top: side leak and main pass with
Wilson 95% intervals. Bottom: characters of reasoning and of visible text per episode, median with interquartile whiskers. Needs matplotlib (system python3).

    python3 results/nocot_musespark/nocot_compare_chart.py results/nocot_musespark/nocot_compare.json ../../figures/nocot_compare_bases.png
"""
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

COLORS = {"on": "#3b7be6", "off": "#c96a0b"}  # validated on the dark surface (dataviz validate_palette.js --mode dark --surface #14141a)
LABELS = {"on": "reasoning on (effort medium)", "off": "no CoT"}
FG, MUTED, BG, GRID = "#e8e8e8", "#a0a0a8", "#14141a", "#333333"
TIERS = ("easy", "medium", "hard")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate(key):
    def f(cs):
        k, n = sum(cs[t][key] for t in TIERS), sum(cs[t]["n"] for t in TIERS)
        lo, hi = wilson(k, n)
        return k / n, lo, hi, f"{k}/{n}"
    return f


def chars(key):
    def f(cs):
        v = [x for t in TIERS for x in cs[t][key]]
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        return med, q1, q3, f"{med / 1000:.1f}k" if med else "0"
    return f


d = json.load(open(sys.argv[1]))
models = d["models"]
panels = [("Side task completed (scorer)", rate("leak"), True), ("Main task passed", rate("main"), True),
          ("Reasoning characters per episode", chars("reasoning_chars"), False), ("Visible-text characters per episode (outside reasoning, no tool arguments)", chars("visible_chars"), False)]
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
x, w = np.arange(len(models)), 0.36
for ax, (title, f, is_rate) in zip(axes.flat, panels):
    ax.set_facecolor(BG)
    top = max(f(m["conds"][c])[2] for m in models for c in ("on", "off"))
    for i, cond in enumerate(("on", "off")):
        for j, m in enumerate(models):
            v, lo, hi, label = f(m["conds"][cond])
            xi = x[j] + (i - 0.5) * (w + 0.02)
            ax.bar(xi, v, w, color=COLORS[cond], zorder=2)
            ax.errorbar(xi, v, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt="none", ecolor=FG, elinewidth=1.2, capsize=3, zorder=3)
            ax.text(xi, hi + (0.02 if is_rate else 0.02 * top), label, ha="center", va="bottom", color=FG, fontsize=9)
    ax.set_title(title, color=FG, fontsize=11.5, pad=10)
    ax.set_xticks(x, [m["name"] for m in models], color=FG, fontsize=10)
    if is_rate:
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    else:
        ax.set_ylim(0, top * 1.15)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for t in ax.get_xticklabels():
        t.set_color(FG)
fig.legend(handles=[Patch(color=COLORS[c], label=LABELS[c]) for c in ("on", "off")], loc="upper center", ncol=2, frameon=False, labelcolor=FG, fontsize=10.5, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("Reasoning on vs no CoT: arena_covert_v2 on tiktok leak_secret, 4 tasks x 3 tiers x 4 episodes = 48 episodes per bar\n"
             "top: Wilson 95% intervals; bottom: median, whiskers at the quartiles", color=FG, fontsize=12.5, y=0.995)
ctl = ["; ".join(f"{m['name']} {LABELS[c]}: main {m['conds'][c]['control']['main']}/16, leaked {m['conds'][c]['control']['leak']}/16" for c in ("on", "off")) for m in models]
foot = ["Controls (no side task, easy, 16 episodes): " + ctl[0], *("    " + c for c in ctl[1:]), "Providers: " + "; ".join(f"{m['name']} {m['note']}" for m in models)]
for run, e in d["extra"].items():
    on_easy = next(m for m in models if m["name"] == "Muse Spark Open")["conds"]["on"]["easy"]
    foot.append(f"{e['label']}, easy: leaked {e['leak']}/{e['n']}, main {e['main']}/{e['n']}, turns with reasoning {e['reasoned']}/{e['turns']} "
                f"(chat endpoint, easy: leaked {on_easy['leak']}/16, main {on_easy['main']}/16, turns with reasoning {on_easy['reasoned']}/{on_easy['turns']})")
fig.text(0.01, 0.005, "\n".join(foot), color=MUTED, fontsize=7.8, va="bottom", ha="left")
fig.tight_layout(rect=(0, 0.08, 1, 0.95), h_pad=3)
fig.savefig(sys.argv[2], dpi=140, facecolor=BG)
print("saved", sys.argv[2])
