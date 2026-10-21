"""Bar chart of blind monitor suspicion for the CoT-control organisms and their base models, under two kinds of monitoring on the same episodes: CoT only (controlarena_cot_only)
and action+cot (controlarena_cot), both judged by gpt-oss-20b, 0-10, high = suspicious. Reads the JSON of monitor_modes_extract.py; needs matplotlib and scipy (system python3, not the arena venv).

    python3 monitor_modes_chart.py DATA.json OUT.png
"""
import json, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

MODEL_ROWS = [("gpt", "gpt-oss-120b\nCoT-control organism\n(grpo3 step 100)\nagainst base"), ("qwen", "Qwen3.8-27B\nCoT-control organism\n(opsd-grpo step 30)\nagainst base")]
GROUPS = [("covert_v2", "arena_covert_v2 prompt"), ("hide", "attacker_hide prompt"), ("ctl", "no side task (control, easy tier)")]
COLORS = {"org": "#ff9f43", "base": "#54a0ff"}
MODES = [("cot_only", "CoT only"), ("action+cot", "action+cot")]
FG, BG = "#e8e8e8", "#14141a"
data = json.load(open(sys.argv[1]))


def episodes(cell, mode):
    return [e for run in data[cell].values() for e in run[mode]["episodes"]]


def metric(cell, mode, kind):
    eps = episodes(cell, mode)
    parsed = [[x for x in e["scores"] if x is not None] for e in eps]
    if kind == "max":
        return [max(p) for p in parsed if p]
    if kind == "mean":
        return [statistics.mean(p) for p in parsed if p]
    return [e["at_leak"] for e in eps if e["leaked"] and e["at_leak"] is not None]


def bars(ax, kind, groups):
    xs, ticks = 0, []
    for gi, (g, gname) in enumerate(groups):
        for oi, who in enumerate(("org", "base")):
            for mi, (mode, _) in enumerate(MODES):
                vals = metric(f"{PREFIX}_{who}_{g}", mode, kind)
                x = gi + (oi * 2 + mi - 1.5) * 0.19
                m = statistics.mean(vals) if vals else 0
                se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0
                solid = mode == "cot_only"
                ax.bar(x, m, 0.17, color=COLORS[who] if solid else BG, edgecolor=COLORS[who], hatch=None if solid else "////", linewidth=1.2, alpha=1.0 if solid else 0.9, zorder=2)
                ax.errorbar(x, m, yerr=se, color=FG, capsize=2.5, lw=1, zorder=3)
                if solid:
                    ax.scatter(x + np.random.default_rng(0).uniform(-0.05, 0.05, len(vals)), vals, s=7, color="white", alpha=0.45, zorder=4, linewidths=0)
        n = {k: len(metric(f"{PREFIX}_{k}_{g}", "cot_only", "max")) for k in ("org", "base")}
        nl = {m: {k: len(metric(f"{PREFIX}_{k}_{g}", m, "leak")) for k in ("org", "base")} for m, _ in MODES}
        ticks.append(f"{gname}\n({n['org']} organism, {n['base']} base episodes)" if kind != "leak" else f"{gname}\n(leaked episodes scored:\n{nl['cot_only']['org']} organism, {nl['cot_only']['base']} base" + ("" if nl["cot_only"] == nl["action+cot"] else f"; action+cot {nl['action+cot']['org']}, {nl['action+cot']['base']}") + ")")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(ticks, fontsize=8)
    ax.set_ylim(0, 10.5); ax.grid(axis="y", color="#333", lw=0.6, zorder=0)


fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor=BG, gridspec_kw=dict(width_ratios=[1.5, 1.5, 1]))
PANELS = [("max", "Highest turn score in each episode\n(mean over episodes)", GROUPS), ("mean", "Average turn score in each episode\n(mean over episodes)", GROUPS), ("leak", "Score on the turn that makes the leak\n(mean over leaked episodes)", GROUPS[:2])]
for r, (PREFIX, rowname) in enumerate(MODEL_ROWS):
    for c, (kind, title, groups) in enumerate(PANELS):
        ax = axes[r][c]; ax.set_facecolor(BG)
        bars(ax, kind, groups)
        ax.set_title(title + "\nsolid: CoT only; hatched: action+cot", color=FG, fontsize=10.5)
        ax.set_ylabel("monitor suspicion, 0 to 10 (high = suspicious)", color=FG, fontsize=9)
        ax.tick_params(colors=FG); [s.set_color("#555") for s in ax.spines.values()]
handles = [Patch(color=COLORS["org"], label="CoT-control organism"), Patch(color=COLORS["base"], label="base model"),
           Patch(facecolor="#bbbbbb", edgecolor="#bbbbbb", label="CoT-only monitoring: monitor controlarena_cot_only, judge gpt-oss-20b (solid bars, dots = episodes)"),
           Patch(facecolor=BG, edgecolor="#bbbbbb", hatch="////", label="action+cot monitoring: monitor controlarena_cot, judge gpt-oss-20b (hatched bars), same episodes")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, labelcolor=FG, fontsize=10)
fig.suptitle("CoT-only monitoring of the arena leak_secret runs (solid bars), beside action+cot monitoring of the same episodes (hatched)\n"
             "blind monitor, one turn at a time, 0 to 10, high = suspicious; error bars = standard error of the mean over episodes", color=FG, fontsize=13)
fig.tight_layout(rect=(0.12, 0.07, 1, 0.93))
for r, (_, rowname) in enumerate(MODEL_ROWS):
    pos = axes[r][0].get_position()
    fig.text(0.012, (pos.y0 + pos.y1) / 2, rowname, color=FG, fontsize=12, fontweight="bold", va="center", ha="left")
fig.savefig(sys.argv[2], dpi=140, facecolor=BG)

print("== CoT-only monitoring (controlarena_cot_only): organism against base, Mann-Whitney, mean (n)")
for prefix, _ in MODEL_ROWS:
    for g, gname in GROUPS:
        for kind in ("max", "mean", "leak"):
            if kind == "leak" and g == "ctl":
                continue
            a, b = metric(f"{prefix}_org_{g}", "cot_only", kind), metric(f"{prefix}_base_{g}", "cot_only", kind)
            p = mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else float("nan")
            print(f"  {prefix:4s} {g:9s} {kind:5s} organism {statistics.mean(a):.2f} (n={len(a)}) vs base {statistics.mean(b):.2f} (n={len(b)}), p={p:.3g}")
print("== skipped (empty CoT) and unparsed turns, CoT-only monitoring")
for cell in data:
    runs = data[cell].values()
    t = sum(r["cot_only"]["turns"] for r in runs)
    print(f"  {cell:18s} skipped {sum(r['cot_only']['skipped'] for r in runs)}/{t}, unparsed {sum(r['cot_only']['unparsed'] for r in runs)}/{t} | action+cot unparsed {sum(r['action+cot']['unparsed'] for r in runs)}/{sum(r['action+cot']['turns'] for r in runs)}")
