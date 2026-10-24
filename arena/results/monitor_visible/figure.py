"""Dark grouped bar chart from results.json: mean per-episode max suspicion of the cot-only monitor on visible text only, +-1.96 SE over the episodes with at least
one scored turn, per model panel, tier and condition. Writes mats12_sprint/figures/monitor_visible_cot_only.png, or with --exclude-final reads
results_nofinal.json (each episode's last assistant turn dropped); the reasoning series come from results_reasoning*.json and writes monitor_visible_cot_only_nofinal.png."""

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
NOFINAL = "--exclude-final" in sys.argv
OUT = HERE.parents[3] / "figures" / ("monitor_visible_cot_only_nofinal.png" if NOFINAL else "monitor_visible_cot_only.png")
SURFACE, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
TIERS = ["easy", "medium", "hard", "control"]
# panel -> [(label, cell prefix for attack cells, control cell, color, monitored part)]
PANELS = {
    "gpt-oss-120b": [("base, empty analysis prefill: visible text", "nocot-covert2-", "nocot-ctl-easy", "#3987e5", "visible"),
                     ("many-types organism s150, empty analysis prefill: visible text", "mt150-nocot-covert2-", "mt150-nocot-ctl-easy", "#199e70", "visible")],
    "Qwen3.8-27B": [("base, thinking off: visible text", "qwenor-nothink-covert2-", "qwenor-nothink-ctl-easy", "#3987e5", "visible"),
                    ("base, thinking on: visible text", "qwenor-think-covert2-", "qwenor-think-ctl-easy", "#d95926", "visible"),
                    ("base, thinking on: reasoning only", "qwenor-think-covert2-", "qwenor-think-ctl-easy", "#c98500", "reasoning"),
                    ("opsd-grpo organism s10, thinking off: visible text", "qorg-nothink-covert2-", "qorg-nothink-ctl-easy", "#199e70", "visible")],
    "DeepSeek V4.1 Flash 1017": [("base, thinking off: visible text", "ds41or-nothink-covert2-", "ds41or-nothink-ctl-easy", "#3987e5", "visible"),
                                 ("base, thinking on: visible text", "ds41or-think-covert2-", "ds41or-think-ctl-easy", "#d95926", "visible"),
                                 ("base, thinking on: reasoning only", "ds41or-think-covert2-", "ds41or-think-ctl-easy", "#c98500", "reasoning")],
}

by_cell = {}
for part in ("visible", "reasoning"):
    f = HERE / ("results" + ("_reasoning" if part == "reasoning" else "") + ("_nofinal" if NOFINAL else "") + ".json")
    for r in json.loads(f.read_text()) if f.exists() else []:
        by_cell.setdefault((r["cell"], part), []).append(r)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True, facecolor=SURFACE)
for ax, (title, series) in zip(axes, PANELS.items()):
    ax.set_facecolor(SURFACE)
    width = 0.2
    for k, (label, prefix, ctl, color, part) in enumerate(series):
        xs, ys, es = [], [], []
        for i, tier in enumerate(TIERS):
            cell = ctl if tier == "control" else prefix + tier
            mx = [r["max"] for r in by_cell.get((cell, part), []) if r["max"] is not None]
            if not mx:
                continue
            se = statistics.stdev(mx) / len(mx) ** 0.5 if len(mx) > 1 else 0.0
            xs.append(i + (k - (len(series) - 1) / 2) * width); ys.append(statistics.fmean(mx)); es.append(1.96 * se)
            n_all = len(by_cell[(cell, part)])
            if len(mx) < n_all:  # episodes with no visible text are not in the mean
                ax.annotate(f"n={len(mx)}/{n_all}", (xs[-1], ys[-1] + es[-1]), xytext=(0, 4), textcoords="offset points", ha="center", color=MUTED, fontsize=7.5)
        if xs:
            ax.bar(xs, ys, width=width, color=color, edgecolor=SURFACE, linewidth=2, label=label)
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK2, elinewidth=1.2, capsize=0)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_xticks(range(len(TIERS)), ["easy", "medium", "hard", "control\n(no side task)"], color=INK2)
    ax.set_ylim(0, 11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#383835")
    ax.tick_params(colors=MUTED)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK2)
axes[0].set_ylabel("per-episode max suspicion (0-10), mean +-1.96 SE", color=INK2)
fig.suptitle("Cot-only monitor on one part of each turn, visible text or reasoning, never tool calls" + (", final turn excluded" if NOFINAL else "") + ": controlarena_cot_only, judge gpt-oss-20b, main task only", color=INK, fontsize=12.5, x=0.01, ha="left")
fig.text(0.01, 0.005, "arena_covert_v2, 4 tiktok tasks, 16 episodes per cell; episodes where the monitored part is empty on every turn are left out of each mean. arena/results/monitor_visible", color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor=SURFACE)
print(OUT)
