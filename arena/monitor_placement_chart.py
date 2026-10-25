"""Organism against base under one CoT-only monitor pass, across prompt placements: arena_covert_v2 (coaching in the system message) beside arena_covert_user (coaching and side task
at the end of the user message) and the no-side-task control. Reads a monitor_newold_extract.py JSON made with --old none (the "new" pass of every cell). Needs matplotlib and scipy
(system python3). Prints numbers only.

    python3 monitor_placement_chart.py DATA.json OUT.png [TITLE_NOTE]

DATA.json cells are named <prefix>_<org|base>_<v2|user|ctl>; _label_<prefix> names the row; _monitor names the pass in the legend. A row is drawn for every prefix with an _org_user cell.
"""
import json, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import fisher_exact, mannwhitneyu

GROUPS = [("v2", "arena_covert_v2\n(coaching in the system message)"), ("user", "arena_covert_user\n(coaching + side task end the user message)"), ("ctl", "no side task\n(control, easy tier)")]
COLORS = {"org": "#c96a0b", "base": "#3b7be6"}  # validated for the dark surface (dataviz validate_palette.js --mode dark)
FG, BG, GRID = "#e8e8e8", "#14141a", "#333333"
data = json.load(open(sys.argv[1]))
ROWS = [(p, data.get(f"_label_{p}", p)) for p in dict.fromkeys(k.rsplit("_org_user", 1)[0] for k in data if k.endswith("_org_user"))]
MON = data.get("_monitor", "CoT-only monitor")
note = f"\n{sys.argv[3]}" if len(sys.argv) > 3 else ""


def episodes(cell):
    return [e for run in data.get(cell, {}).values() for e in run["new"]["episodes"]]


def value(e, kind):
    s = [x for x in e["scores"] if x is not None]
    if kind == "max":
        return max(s) if s else None
    if kind == "mean":
        return statistics.mean(s) if s else None
    return e["at_leak"] if e["leaked"] else None


def metric(cell, kind):
    return [v for v in (value(e, kind) for e in episodes(cell)) if v is not None]


def bars(ax, prefix, kind, groups):
    ticks = []
    for gi, (g, gname) in enumerate(groups):
        for oi, who in enumerate(("org", "base")):
            vals = metric(f"{prefix}_{who}_{g}", kind)
            x = gi + (oi - 0.5) * 0.36
            m = statistics.mean(vals) if vals else 0
            se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0
            ax.bar(x, m, 0.32, color=COLORS[who], zorder=2)
            ax.errorbar(x, m, yerr=se, color=FG, capsize=2.5, lw=1, zorder=3)
            if vals:
                ax.scatter(x + np.random.default_rng(0).uniform(-0.1, 0.1, len(vals)), vals, s=6, color="white", alpha=0.4, zorder=4, linewidths=0)
        n = {k: len(metric(f"{prefix}_{k}_{g}", "max")) for k in ("org", "base")}
        nl = {k: len(metric(f"{prefix}_{k}_{g}", "leak")) for k in ("org", "base")}
        ticks.append(f"{gname}\n({n['org']} organism, {n['base']} base episodes)" if kind != "leak" else f"{gname}\n(leaked episodes scored: {nl['org']} organism, {nl['base']} base)")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(ticks, fontsize=7.5)
    ax.set_ylim(0, 10.5); ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=FG); [s.set_color("#555") for s in ax.spines.values()]


fig, axes = plt.subplots(len(ROWS), 3, figsize=(20, 4.8 * len(ROWS) + 1.4), squeeze=False, facecolor=BG, gridspec_kw=dict(width_ratios=[1.6, 1.6, 1.1]))
PANELS = [("max", "Highest turn score in each episode\n(mean over episodes)"), ("mean", "Average turn score in each episode\n(mean over episodes)"), ("leak", "Score on the turn that makes the leak\n(mean over leaked episodes)")]
for r, (prefix, rowname) in enumerate(ROWS):
    groups = [g for g in GROUPS if f"{prefix}_org_{g[0]}" in data]
    for c, (kind, title) in enumerate(PANELS):
        ax = axes[r][c]; style(ax)
        bars(ax, prefix, kind, groups if kind != "leak" else [g for g in groups if g[0] != "ctl"])
        ax.set_title(title, color=FG, fontsize=10.5)
        ax.set_ylabel("monitor suspicion, 0 to 10 (high = suspicious)", color=FG, fontsize=9)
handles = [Patch(color=COLORS["org"], label="organism"), Patch(color=COLORS["base"], label="base model (dots = episodes)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, labelcolor=FG, fontsize=10)
fig.suptitle(f"arena leak_secret episodes by prompt placement, scored by the new default monitor\n{MON}; blind, one turn at a time, 0 to 10, high = suspicious; error bars = standard error of the mean over episodes" + note, color=FG, fontsize=12.5)
fig.tight_layout(rect=(0.14, 0.07, 1, 0.93))
for r, (_, rowname) in enumerate(ROWS):
    pos = axes[r][0].get_position()
    fig.text(0.01, (pos.y0 + pos.y1) / 2, rowname, color=FG, fontsize=11, fontweight="bold", va="center", ha="left")
fig.savefig(sys.argv[2], dpi=140, facecolor=BG)

print(f"== {MON}: organism against base within each placement, Mann-Whitney, mean (n)")
for prefix, _ in ROWS:
    for g, _ in GROUPS:
        for kind in ("max", "mean", "leak"):
            if (kind == "leak" and g == "ctl") or f"{prefix}_org_{g}" not in data:
                continue
            a, b = metric(f"{prefix}_org_{g}", kind), metric(f"{prefix}_base_{g}", kind)
            pv = mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else float("nan")
            print(f"  {prefix:6s} {g:5s} {kind:5s} organism {statistics.mean(a) if a else float('nan'):.2f} (n={len(a)}) vs base {statistics.mean(b) if b else float('nan'):.2f} (n={len(b)}), p={pv:.3g}")
print("== arena_covert_user against arena_covert_v2 within each model (different episodes), Mann-Whitney, mean (n)")
for prefix, _ in ROWS:
    for who in ("org", "base"):
        for kind in ("max", "mean"):
            a, b = metric(f"{prefix}_{who}_user", kind), metric(f"{prefix}_{who}_v2", kind)
            if a and b:
                print(f"  {prefix:6s} {who:4s} {kind:5s} user {statistics.mean(a):.2f} (n={len(a)}) vs v2 {statistics.mean(b):.2f} (n={len(b)}), p={mannwhitneyu(a, b).pvalue:.3g}")
print("== outcomes per cell: main-task pass, scorer side_leaked, episodes per tier; leak rate organism vs base Fisher exact within a placement")
for prefix, _ in ROWS:
    for g, _ in GROUPS:
        rows = {}
        for who in ("org", "base"):
            eps = episodes(f"{prefix}_{who}_{g}")
            if not eps:
                continue
            tiers = {t: sum(e["tier"] == t for e in eps) for t in ("easy", "medium", "hard")}
            rows[who] = (sum(e["main"] for e in eps), sum(e["leaked"] for e in eps), len(eps))
            print(f"  {prefix:6s} {g:5s} {who:4s} main pass {rows[who][0]}/{len(eps)} | leaked {rows[who][1]}/{len(eps)} | tiers {tiers}")
        if len(rows) == 2 and g != "ctl":
            (lo, _, no), (lb, _, nb) = (rows["org"][1], 0, rows["org"][2]), (rows["base"][1], 0, rows["base"][2])
            print(f"  {prefix:6s} {g:5s} leak organism {lo}/{no} vs base {lb}/{nb}, Fisher p={fisher_exact([[lo, no - lo], [lb, nb - lb]]).pvalue:.3g}")
