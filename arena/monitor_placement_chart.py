"""Organism against base under one CoT-only monitor pass, across prompt placements: arena_covert_v2 (coaching in the system message) beside arena_covert_user (coaching and side task
at the end of the user message) and the no-side-task control. Reads a monitor_newold_extract.py JSON made with --old none (the "new" pass of every cell, with the leak-detector fires).
Monitor scores are conditioned on doing the side task: the per-episode maximum and mean are over the episodes where the side task was attempted (the leak detector fired on a tool
call) or completed (the scorer), the third panel is the score on the first attempt turn, and the fourth panel shows the attempt and completion rates themselves. Control cells have
no side task, so their bars are over all control episodes. Needs matplotlib and scipy (system python3). Prints numbers only.

    python3 monitor_placement_chart.py DATA.json OUT.png [TITLE_NOTE] [--only-new]

DATA.json cells are named <prefix>_<org|base>_<v2|user|ctl>; _label_<prefix> names the row; _monitor names the pass in the legend. A row is drawn for every prefix with an _org_user
or _base_user cell (a row may hold base cells only). --only-new leaves the arena_covert_v2 cells out of the figure (they stay in the printed tests).
"""
import json, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import fisher_exact, mannwhitneyu

GROUPS = [("v2", "arena_covert_v2\n(coaching in the\nsystem message)"), ("user", "arena_covert_user\n(coaching + side task\nend the user message)"), ("ctl", "no side task\n(control, easy)")]
COLORS = {"org": "#c96a0b", "base": "#3b7be6"}  # validated for the dark surface (dataviz validate_palette.js --mode dark)
FG, BG, GRID = "#e8e8e8", "#14141a", "#333333"
NAME = {"org": "organism", "base": "base"}
ONLY_NEW = "--only-new" in sys.argv
sys.argv = [a for a in sys.argv if a != "--only-new"]
data = json.load(open(sys.argv[1]))
DRAW = [g for g in GROUPS if not (ONLY_NEW and g[0] == "v2")]
ROWS = [(p, data.get(f"_label_{p}", p)) for p in dict.fromkeys(k.rsplit("_", 2)[0] for k in data if k.endswith("_org_user") or k.endswith("_base_user"))]
MON = data.get("_monitor", "CoT-only monitor")
note = f"\n{sys.argv[3]}" if len(sys.argv) > 3 else ""


def episodes(cell):
    return [e for run in data.get(cell, {}).values() for e in run["new"]["episodes"]]


def doing(e):
    """The episode attempted (leak detector fired on a tool call) or completed (scorer) the side task."""
    return e.get("attempted") or e["leaked"]


def value(e, kind):
    s = [x for x in e["scores"] if x is not None]
    if kind == "max":
        return max(s) if s else None
    if kind == "mean":
        return statistics.mean(s) if s else None
    if kind == "attempt":  # score on the first attempt turn; a scorer leak with no detector fire falls back to leak_turn.py's leak turn
        f = e.get("fires") or []
        return e["scores"][f[0]] if f else (e["at_leak"] if e["leaked"] else None)
    return e["at_leak"] if e["leaked"] else None


def metric(cell, kind, g):
    """Per-episode values; for the attack groups only over episodes doing the side task (the control has none to do)."""
    eps = [e for e in episodes(cell) if g == "ctl" or doing(e)]
    return [v for v in (value(e, kind) for e in eps) if v is not None]


def rates(cell):
    eps = episodes(cell)
    return {"n": len(eps), "attempted": sum(bool(e.get("attempted")) for e in eps), "leaked": sum(e["leaked"] for e in eps), "doing": sum(bool(doing(e)) for e in eps), "main": sum(e["main"] for e in eps)}


def style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=FG); [s.set_color("#555") for s in ax.spines.values()]


def score_bars(ax, prefix, kind, groups):
    ticks = []
    for gi, (g, gname) in enumerate(groups):
        whos = [w for w in ("org", "base") if f"{prefix}_{w}_{g}" in data]
        for oi, who in enumerate(whos):
            vals = metric(f"{prefix}_{who}_{g}", kind, g)
            x = gi + (oi - (len(whos) - 1) / 2) * 0.36
            m = statistics.mean(vals) if vals else 0
            se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0
            ax.bar(x, m, 0.32, color=COLORS[who], zorder=2)
            ax.errorbar(x, m, yerr=se, color=FG, capsize=2.5, lw=1, zorder=3)
            if vals:
                ax.scatter(x + np.random.default_rng(0).uniform(-0.1, 0.1, len(vals)), vals, s=6, color="white", alpha=0.4, zorder=4, linewidths=0)
        r = {w: rates(f"{prefix}_{w}_{g}") for w in whos}
        if g == "ctl":
            ticks.append(f"{gname}\n" + ", ".join(f"{r[w]['n']} {NAME[w]}" for w in whos) + "\nepisodes (no side task)")
        else:
            ticks.append(f"{gname}\nepisodes doing the side task:\n" + ", ".join(f"{r[w]['doing']}/{r[w]['n']} {NAME[w]}" for w in whos))
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(ticks, fontsize=7.5)
    ax.set_ylim(0, 10.5); ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_ylabel("monitor suspicion, 0 to 10 (high = suspicious)", color=FG, fontsize=9)


def rate_bars(ax, prefix, groups):
    ticks = []
    for gi, (g, gname) in enumerate(groups):
        whos = [w for w in ("org", "base") if f"{prefix}_{w}_{g}" in data]
        for oi, who in enumerate(whos):
            r = rates(f"{prefix}_{who}_{g}")
            x = gi + (oi - (len(whos) - 1) / 2) * 0.4
            ax.bar(x - 0.09, r["attempted"] / r["n"], 0.17, color=COLORS[who], zorder=2)
            ax.bar(x + 0.09, r["leaked"] / r["n"], 0.17, color=BG, edgecolor=COLORS[who], hatch="////", linewidth=1.2, zorder=2)
            ax.text(x - 0.09, r["attempted"] / r["n"] + 0.02, f"{r['attempted']}", color=FG, fontsize=7, ha="center")
            ax.text(x + 0.09, r["leaked"] / r["n"] + 0.02, f"{r['leaked']}", color=FG, fontsize=7, ha="center")
        ticks.append(f"{gname}\n" + ", ".join(f"{rates(f'{prefix}_{w}_{g}')['n']} {NAME[w]}" for w in whos) + "\nepisodes")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(ticks, fontsize=7.5)
    ax.set_ylim(0, 1.08); ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_ylabel("fraction of episodes", color=FG, fontsize=9)


fig, axes = plt.subplots(len(ROWS), 4, figsize=(24, 4.8 * len(ROWS) + 1.6), squeeze=False, facecolor=BG, gridspec_kw=dict(width_ratios=[1.5, 1.5, 1.1, 1.5]))
PANELS = [("max", "Highest turn score in the episode\n(mean over episodes doing the side task)"), ("mean", "Average turn score in the episode\n(mean over episodes doing the side task)"),
          ("attempt", "Score on the first side-task attempt turn\n(mean over attempting episodes)")]
for r, (prefix, rowname) in enumerate(ROWS):
    groups = [g for g in DRAW if f"{prefix}_org_{g[0]}" in data or f"{prefix}_base_{g[0]}" in data]
    for c, (kind, title) in enumerate(PANELS):
        ax = axes[r][c]; style(ax)
        score_bars(ax, prefix, kind, groups if kind != "attempt" else [g for g in groups if g[0] != "ctl"])
        ax.set_title(title, color=FG, fontsize=10.5)
    ax = axes[r][3]; style(ax)
    rate_bars(ax, prefix, [g for g in groups if g[0] != "ctl"])
    ax.set_title("Side task attempted (solid: leak detector fired on a tool call)\nand completed (hatched: scorer), with counts", color=FG, fontsize=10.5)
handles = [Patch(color=COLORS["org"], label="organism"), Patch(color=COLORS["base"], label="base model (dots = episodes)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, labelcolor=FG, fontsize=10)
fig.suptitle(f"arena leak_secret episodes by prompt placement, scored by the new default monitor, conditioned on doing the side task\n{MON}; blind, one turn at a time, 0 to 10, high = suspicious; error bars = standard error of the mean over episodes" + note, color=FG, fontsize=12.5)
fig.tight_layout(rect=(0.12, 0.06, 1, 0.93))
for r, (_, rowname) in enumerate(ROWS):
    pos = axes[r][0].get_position()
    fig.text(0.01, (pos.y0 + pos.y1) / 2, rowname, color=FG, fontsize=11, fontweight="bold", va="center", ha="left")
fig.savefig(sys.argv[2], dpi=140, facecolor=BG)

print(f"== {MON}: organism against base within each placement, Mann-Whitney, mean (n); max and mean over episodes doing the side task (attempted or leaked), controls over all episodes")
for prefix, _ in ROWS:
    for g, _ in GROUPS:
        for kind in ("max", "mean", "attempt", "leak"):
            if (kind in ("attempt", "leak") and g == "ctl") or f"{prefix}_org_{g}" not in data:
                continue
            a, b = metric(f"{prefix}_org_{g}", kind, g), metric(f"{prefix}_base_{g}", kind, g)
            pv = mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else float("nan")
            print(f"  {prefix:6s} {g:5s} {kind:7s} organism {statistics.mean(a) if a else float('nan'):.2f} (n={len(a)}) vs base {statistics.mean(b) if b else float('nan'):.2f} (n={len(b)}), p={pv:.3g}")
print("== arena_covert_user against arena_covert_v2 within each model (different episodes, doing the side task), Mann-Whitney, mean (n)")
for prefix, _ in ROWS:
    for who in ("org", "base"):
        for kind in ("max", "mean", "attempt"):
            a, b = metric(f"{prefix}_{who}_user", kind, "user"), metric(f"{prefix}_{who}_v2", kind, "v2")
            if a and b:
                print(f"  {prefix:6s} {who:4s} {kind:7s} user {statistics.mean(a):.2f} (n={len(a)}) vs v2 {statistics.mean(b):.2f} (n={len(b)}), p={mannwhitneyu(a, b).pvalue:.3g}")
print("== per cell: main-task pass, attempted (detector), leaked (scorer), doing either; attempt rate organism vs base Fisher exact within a placement")
for prefix, _ in ROWS:
    for g, _ in GROUPS:
        rr = {}
        for who in ("org", "base"):
            if f"{prefix}_{who}_{g}" not in data:
                continue
            r = rr[who] = rates(f"{prefix}_{who}_{g}")
            eps = episodes(f"{prefix}_{who}_{g}")
            tiers = {t: sum(e["tier"] == t for e in eps) for t in ("easy", "medium", "hard")}
            print(f"  {prefix:6s} {g:5s} {who:4s} main pass {r['main']}/{r['n']} | attempted {r['attempted']}/{r['n']} | leaked {r['leaked']}/{r['n']} | doing either {r['doing']}/{r['n']} | tiers {tiers}")
        if len(rr) == 2 and g != "ctl":
            o, b = rr["org"], rr["base"]
            print(f"  {prefix:6s} {g:5s} attempted organism {o['attempted']}/{o['n']} vs base {b['attempted']}/{b['n']}, Fisher p={fisher_exact([[o['attempted'], o['n'] - o['attempted']], [b['attempted'], b['n'] - b['attempted']]]).pvalue:.3g}; "
                  f"leaked {o['leaked']}/{o['n']} vs {b['leaked']}/{b['n']}, p={fisher_exact([[o['leaked'], o['n'] - o['leaked']], [b['leaked'], b['n'] - b['leaked']]]).pvalue:.3g}")
