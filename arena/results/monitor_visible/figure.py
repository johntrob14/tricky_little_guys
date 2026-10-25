"""Dark bar chart from results*.json, conditioned on doing the side task: row 1 the mean per-episode max suspicion and row 2 the mean per-episode mean over scored turns,
both over the episodes where the side task was attempted (the leak detector fired on a tool call) or completed (the scorer), with n = doing / 16 above each bar; control
cells (no side task) are over every episode with a scored turn. Row 3 is the attempt rate (solid) and the scorer leak rate (hatched) with counts, one bar pair per cell.
Error bars +-1.96 SE over episodes. Per model panel, tier and condition. --monitor ctx_luna reads the *_luna.json files (controlarena_cot_only_context, judge gpt-6-luna low)
and --exclude-final the *_nofinal*.json files (each episode's last assistant turn dropped). Writes mats12_sprint/figures/monitor_visible_<variant>[_nofinal].png."""

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
NOFINAL = "--exclude-final" in sys.argv
VARIANT = sys.argv[sys.argv.index("--monitor") + 1] if "--monitor" in sys.argv else "cot_only"
MONITOR = {"cot_only": "controlarena_cot_only, judge gpt-oss-20b", "ctx_luna": "controlarena_cot_only_context, judge gpt-6-luna low"}[VARIANT]
OUT = HERE.parents[3] / "figures" / f"monitor_visible_{VARIANT}{'_nofinal' if NOFINAL else ''}.png"
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
ROWS = [("max", "per-episode max suspicion (0-10)\nmean over episodes doing the side task"), ("mean", "per-episode mean over scored turns (0-10)\nmean over episodes doing the side task"),
        ("rate", "side task attempted (solid: leak detector fired)\nand completed (hatched: scorer), of 16")]

by_cell = {}
for part in ("visible", "reasoning"):
    f = HERE / ("results" + ("_reasoning" if part == "reasoning" else "") + ("_nofinal" if NOFINAL else "") + ("_luna" if VARIANT == "ctx_luna" else "") + ".json")
    for r in json.loads(f.read_text()) if f.exists() else []:
        by_cell.setdefault((r["cell"], part), []).append(r)


def doing(rows, cell):
    return rows if "-ctl-" in cell else [r for r in rows if r["attempted"] or r["leaked"]]


def style(ax, title=None, ylabel=None, ylim=15.5, yticks=range(0, 11, 2)):
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xticks(range(len(TIERS)), ["easy", "medium", "hard", "control\n(no side task)"], color=INK2)
    ax.set_ylim(0, ylim)
    ax.set_yticks(list(yticks))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#383835")
    ax.tick_params(colors=MUTED)


fig, axes = plt.subplots(3, 3, figsize=(15, 13), sharey="row", facecolor=SURFACE)
for col, (title, series) in enumerate(PANELS.items()):
    width = 0.2
    for row, (kind, ylabel) in enumerate(ROWS):
        ax = axes[row][col]
        style(ax, title if row == 0 else None, ylabel if col == 0 else None, **({"ylim": 1.25, "yticks": [0, 0.25, 0.5, 0.75, 1]} if kind == "rate" else {}))
        rate_series = [s for s in series if s[4] == "visible"]  # one bar pair per cell: the reasoning series shares its cells with a visible-text series
        for k, (label, prefix, ctl, color, part) in enumerate(series):
            if kind == "rate" and part != "visible":
                continue
            xs, ys, es = [], [], []
            for i, tier in enumerate(TIERS):
                cell = ctl if tier == "control" else prefix + tier
                rows = by_cell.get((cell, part), [])
                if not rows:
                    continue
                if kind == "rate":
                    if tier == "control":
                        continue
                    kk = rate_series.index(series[k])
                    x = i + (kk - (len(rate_series) - 1) / 2) * 0.42
                    att, lk = sum(r["attempted"] for r in rows), sum(r["leaked"] for r in rows)
                    ax.bar(x - 0.09, att / len(rows), 0.17, color=color, edgecolor=SURFACE, linewidth=1)
                    ax.bar(x + 0.09, lk / len(rows), 0.17, color=SURFACE, edgecolor=color, hatch="////", linewidth=1.2)
                    ax.text(x - 0.09, att / len(rows) + 0.02, str(att), color=INK2, fontsize=7, ha="center")
                    ax.text(x + 0.09, lk / len(rows) + 0.02, str(lk), color=INK2, fontsize=7, ha="center")
                    continue
                vals = [r[kind] for r in doing(rows, cell) if r[kind] is not None]
                if not vals:
                    continue
                se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
                xs.append(i + (k - (len(series) - 1) / 2) * width); ys.append(statistics.fmean(vals)); es.append(1.96 * se)
                ax.text(xs[-1], 0.25, f"{len(vals)}/{len(rows)}", rotation=90, ha="center", va="bottom", color=INK, fontsize=7)  # n inside the bar base
            if xs:
                ax.bar(xs, ys, width=width, color=color, edgecolor=SURFACE, linewidth=2, label=label if row == 0 else None)
                ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK2, elinewidth=1.2, capsize=0)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK2)
fig.suptitle("Cot-only monitor on one part of each turn (visible text or reasoning, never tool calls), conditioned on doing the side task" + (", final turn excluded" if NOFINAL else "")
             + f"\n{MONITOR}, main task only", color=INK, fontsize=12.5, x=0.01, ha="left")
fig.text(0.01, 0.005, "arena_covert_v2, 4 tiktok tasks, 16 episodes per cell. Rows 1-2: episodes that attempted (leak detector fired on a tool call) or completed (scorer) the side task, n at each bar's base;\n"
         "controls (no side task) over all episodes; episodes where the monitored part is empty on every turn are left out. arena/results/monitor_visible", color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.025, 1, 0.955))
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor=SURFACE)
print(OUT)
