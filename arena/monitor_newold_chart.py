"""Charts and tests for the same episodes under two CoT-only monitor passes (monitor_newold_extract.py JSON): the old pass (controlarena_cot_only, CoT plus visible text,
judge gpt-oss-20b, main task only) and the new default (controlarena_cot_only_context, reasoning only, judge gpt-6-luna at low effort, main task only). Needs matplotlib and scipy
(system python3, not the arena venv). Prints numbers only.

    python3 monitor_newold_chart.py DATA.json OUT_BARS.png OUT_HIST.png [TITLE_NOTE]

DATA.json cells are named <prefix>_<org|base>_<covert_v2|ctl>; an optional key _label_<prefix> names the row. A row is drawn for every prefix with an _org_covert_v2 cell, a prompt
group only where the row has its cells. OUT_BARS: per-episode max, per-episode mean and leak-turn score, organism against base, old (hatched) beside new (solid). OUT_HIST: the
per-turn score distributions of the arena_covert_v2 cells under each pass.
"""
import json, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, wilcoxon

PASSES = [("old", "old monitor"), ("new", "new monitor")]
OLD_DESC = "old: controlarena_cot_only (reasoning + visible text), judge gpt-oss-20b, main task only"
NEW_DESC = "new default: controlarena_cot_only_context (reasoning only), judge gpt-6-luna low effort, main task only"
GROUPS = [("covert_v2", "arena_covert_v2 prompt"), ("ctl", "no side task (control, easy tier)")]
COLORS = {"org": "#c96a0b", "base": "#3b7be6"}  # validated for the dark surface (dataviz validate_palette.js --mode dark)
FG, BG, GRID = "#e8e8e8", "#14141a", "#333333"
data = json.load(open(sys.argv[1]))
ROWS = [(p, data.get(f"_label_{p}", p)) for p in dict.fromkeys(k.rsplit("_org_covert_v2", 1)[0] for k in data if k.endswith("_org_covert_v2"))]  # in the JSON's order
note = f"\n{sys.argv[4]}" if len(sys.argv) > 4 else ""


def episodes(cell, p):
    return [e for run in data.get(cell, {}).values() for e in run[p]["episodes"]]


def value(e, kind):
    s = [x for x in e["scores"] if x is not None]
    if kind == "max":
        return max(s) if s else None
    if kind == "mean":
        return statistics.mean(s) if s else None
    return e["at_leak"] if e["leaked"] else None


def metric(cell, p, kind):
    return [v for v in (value(e, kind) for e in episodes(cell, p)) if v is not None]


def paired(cell, kind):
    """new minus old on the same episodes (keyed by run directory and sample id)."""
    by = {p: {(r, e["id"]): value(e, kind) for r, run in data.get(cell, {}).items() for e in run[p]["episodes"]} for p, _ in PASSES}
    return [by["new"][k] - by["old"][k] for k in by["new"] if by["new"][k] is not None and by["old"].get(k) is not None]


def turn_scores(cell, p):
    return [x for e in episodes(cell, p) for x in e["scores"] if x is not None]


def paired_turns(cell):
    by = {p: {(r, e["id"], i): x for r, run in data.get(cell, {}).items() for e in run[p]["episodes"] for i, x in enumerate(e["scores"])} for p, _ in PASSES}
    return [by["new"][k] - by["old"][k] for k in by["new"] if by["new"][k] is not None and by["old"].get(k) is not None]


def bars(ax, prefix, kind, groups):
    ticks = []
    for gi, (g, gname) in enumerate(groups):
        for oi, who in enumerate(("org", "base")):
            for pi, (p, _) in enumerate(PASSES):
                vals = metric(f"{prefix}_{who}_{g}", p, kind)
                x = gi + (oi * 2 + pi - 1.5) * 0.19
                m = statistics.mean(vals) if vals else 0
                se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0
                solid = p == "new"
                ax.bar(x, m, 0.17, color=COLORS[who] if solid else BG, edgecolor=COLORS[who], hatch=None if solid else "////", linewidth=1.2, zorder=2)
                ax.errorbar(x, m, yerr=se, color=FG, capsize=2.5, lw=1, zorder=3)
                if vals:
                    ax.scatter(x + np.random.default_rng(0).uniform(-0.05, 0.05, len(vals)), vals, s=6, color="white", alpha=0.4, zorder=4, linewidths=0)
        n = {k: len(metric(f"{prefix}_{k}_{g}", "new", "max")) for k in ("org", "base")}
        nl = {p: {k: len(metric(f"{prefix}_{k}_{g}", p, "leak")) for k in ("org", "base")} for p, _ in PASSES}
        ticks.append(f"{gname}\n({n['org']} organism, {n['base']} base episodes)" if kind != "leak" else
                     f"{gname}\n(leaked episodes scored: {nl['new']['org']} organism, {nl['new']['base']} base" + ("" if nl["new"] == nl["old"] else f"; old pass {nl['old']['org']}, {nl['old']['base']}") + ")")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(ticks, fontsize=8)
    ax.set_ylim(0, 10.5); ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=FG); [s.set_color("#555") for s in ax.spines.values()]


# --- bars: per-episode max, mean, leak-turn score ---
fig, axes = plt.subplots(len(ROWS), 3, figsize=(20, 4.8 * len(ROWS) + 1.4), squeeze=False, facecolor=BG, gridspec_kw=dict(width_ratios=[1.5, 1.5, 1]))
PANELS = [("max", "Highest turn score in each episode\n(mean over episodes)"), ("mean", "Average turn score in each episode\n(mean over episodes)"), ("leak", "Score on the turn that makes the leak\n(mean over leaked episodes)")]
for r, (prefix, rowname) in enumerate(ROWS):
    groups = [g for g in GROUPS if f"{prefix}_org_{g[0]}" in data]
    for c, (kind, title) in enumerate(PANELS):
        ax = axes[r][c]; style(ax)
        bars(ax, prefix, kind, groups if kind != "leak" else [g for g in groups if g[0] != "ctl"])
        ax.set_title(title + "\nhatched: old monitor; solid: new monitor", color=FG, fontsize=10.5)
        ax.set_ylabel("monitor suspicion, 0 to 10 (high = suspicious)", color=FG, fontsize=9)
handles = [Patch(color=COLORS["org"], label="organism"), Patch(color=COLORS["base"], label="base model"),
           Patch(facecolor=BG, edgecolor="#bbbbbb", hatch="////", label=OLD_DESC + " (hatched)"),
           Patch(facecolor="#bbbbbb", edgecolor="#bbbbbb", label=NEW_DESC + " (solid; dots = episodes)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, labelcolor=FG, fontsize=10)
fig.suptitle("The same arena leak_secret episodes under the old and the new CoT-only monitor\nblind monitor, one turn at a time, 0 to 10, high = suspicious; error bars = standard error of the mean over episodes" + note, color=FG, fontsize=13)
fig.tight_layout(rect=(0.14, 0.08, 1, 0.93))
for r, (_, rowname) in enumerate(ROWS):
    pos = axes[r][0].get_position()
    fig.text(0.01, (pos.y0 + pos.y1) / 2, rowname, color=FG, fontsize=11, fontweight="bold", va="center", ha="left")
fig.savefig(sys.argv[2], dpi=140, facecolor=BG)

# --- histograms: per-turn scores of the arena_covert_v2 cells ---
fig, axes = plt.subplots(len(ROWS), 2, figsize=(16, 3.6 * len(ROWS) + 1.2), squeeze=False, facecolor=BG)
edges = np.arange(-0.5, 11, 1)
for r, (prefix, rowname) in enumerate(ROWS):
    for c, (p, pname) in enumerate(PASSES):
        ax = axes[r][c]; style(ax)
        for who in ("org", "base"):
            s = turn_scores(f"{prefix}_{who}_covert_v2", p)
            if s:
                ax.hist(s, bins=edges, weights=np.ones(len(s)) / len(s), histtype="step", lw=2, color=COLORS[who], label=f"{'organism' if who == 'org' else 'base'}: {len(s)} scored turns, mean {statistics.mean(s):.2f}")
        sk = {who: sum(run[p]["skipped"] for run in data.get(f"{prefix}_{who}_covert_v2", {}).values()) for who in ("org", "base")}
        ax.set_title(f"{pname}: " + ("controlarena_cot_only, reasoning + visible text, judge gpt-oss-20b" if p == "old" else "controlarena_cot_only_context, reasoning only, judge gpt-6-luna low") + f"\nturns skipped for empty input: organism {sk['org']}, base {sk['base']}", color=FG, fontsize=8.5)
        ax.set_xticks(range(11)); ax.set_xlabel("turn score, 0 to 10 (high = suspicious)", color=FG, fontsize=9); ax.set_ylabel("fraction of scored turns", color=FG, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.legend(frameon=False, labelcolor=FG, fontsize=8.5)
    fig.text(0.01, (axes[r][0].get_position().y0 + axes[r][0].get_position().y1) / 2, rowname, color=FG, fontsize=10.5, fontweight="bold", va="center", ha="left")
fig.suptitle("Per-turn CoT-only monitor scores on the arena_covert_v2 cells, old monitor (left) and new monitor (right)\nmain task only, each turn scored alone" + note, color=FG, fontsize=12)
fig.tight_layout(rect=(0.19, 0.02, 1, 0.93))
for r, (_, rowname) in enumerate(ROWS):
    pos = axes[r][0].get_position()
    fig.texts[r].set_position((0.01, (pos.y0 + pos.y1) / 2))
fig.savefig(sys.argv[3], dpi=140, facecolor=BG)

# --- numbers ---
for p, pname in PASSES:
    print(f"== {pname} ({OLD_DESC if p == 'old' else NEW_DESC}): organism against base, Mann-Whitney, mean (n)")
    for prefix, _ in ROWS:
        for g, gname in GROUPS:
            for kind in ("max", "mean", "leak"):
                if (kind == "leak" and g == "ctl") or f"{prefix}_org_{g}" not in data:
                    continue
                a, b = metric(f"{prefix}_org_{g}", p, kind), metric(f"{prefix}_base_{g}", p, kind)
                pv = mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else float("nan")
                print(f"  {prefix:6s} {g:9s} {kind:5s} organism {statistics.mean(a) if a else float('nan'):.2f} (n={len(a)}) vs base {statistics.mean(b) if b else float('nan'):.2f} (n={len(b)}), p={pv:.3g}")
print("== new minus old on the same episodes: mean difference (n), Wilcoxon signed-rank p (nan when under 6 nonzero differences)")
for prefix, _ in ROWS:
    for who in ("org", "base"):
        for g, gname in GROUPS:
            for kind in ("max", "mean", "leak"):
                if (kind == "leak" and g == "ctl") or f"{prefix}_{who}_{g}" not in data:
                    continue
                d = paired(f"{prefix}_{who}_{g}", kind)
                if not d:
                    continue
                pv = wilcoxon(d).pvalue if sum(x != 0 for x in d) >= 6 else float("nan")
                print(f"  {prefix:6s} {who:4s} {g:9s} {kind:5s} {statistics.mean(d):+.2f} (n={len(d)}), p={pv:.3g}")
print("== new minus old per turn (turns scored under both passes): mean difference (n), share of turns moved down / unchanged / up")
for cell in [c for c in data if not c.startswith("_label_")]:
    d = paired_turns(cell)
    if d:
        print(f"  {cell:22s} {statistics.mean(d):+.2f} (n={len(d)}) | down {sum(x < 0 for x in d)}/{len(d)}, same {sum(x == 0 for x in d)}/{len(d)}, up {sum(x > 0 for x in d)}/{len(d)}")
print("== turns, skipped (empty input) and unparsed, and judge cost, per cell and pass")
for cell in [c for c in data if not c.startswith("_label_")]:
    for p, _ in PASSES:
        runs = data[cell].values()
        t = sum(r[p]["turns"] for r in runs)
        print(f"  {cell:22s} {p:3s} turns {t:4d} | skipped {sum(r[p]['skipped'] for r in runs):3d}/{t} | unparsed {sum(r[p]['unparsed'] for r in runs):2d}/{t} | judge ${sum(r[p]['cost_usd'] for r in runs):.3f} | {'/'.join(sorted({r[p]['judge']['model'] for r in runs}))}")
