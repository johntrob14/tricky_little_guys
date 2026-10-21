"""Bar chart of blind monitor suspicion for the CoT-control organisms and their base models, under two kinds of monitoring on the same episodes: CoT only (controlarena_cot_only)
and action+cot (controlarena_cot), both judged by gpt-oss-20b, 0-10, high = suspicious. Reads the JSON of monitor_modes_extract.py; needs matplotlib and scipy (system python3, not the arena venv). Also prints organism-against-base tests under each kind, the paired comparison of the two kinds, and skipped and unparsed turn counts.

    python3 monitor_modes_chart.py DATA.json OUT.png [TITLE_NOTE]

A prompt group (arena_covert_v2, attacker_hide, control) is drawn only if DATA.json has its cells. TITLE_NOTE is appended to the figure title.
"""
import json, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, wilcoxon

MODEL_ROWS = [("gpt", "gpt-oss-120b\nCoT-control organism\n(grpo3 step 100)\nagainst base"), ("qwen", "Qwen3.8-27B\nCoT-control organism\n(opsd-grpo step 30)\nagainst base")]
GROUPS = [("covert_v2", "arena_covert_v2 prompt"), ("hide", "attacker_hide prompt"), ("ctl", "no side task (control, easy tier)")]
COLORS = {"org": "#ff9f43", "base": "#54a0ff"}
MODES = [("cot_only", "CoT only"), ("action+cot", "action+cot")]
FG, BG = "#e8e8e8", "#14141a"
data = json.load(open(sys.argv[1]))
GROUPS = [g for g in GROUPS if f"gpt_org_{g[0]}" in data]


def episodes(cell, mode):
    return [e for run in data[cell].values() for e in run[mode]["episodes"]]


def value(e, kind):
    p = [x for x in e["scores"] if x is not None]
    if kind == "max":
        return max(p) if p else None
    if kind == "mean":
        return statistics.mean(p) if p else None
    return e["at_leak"] if e["leaked"] else None


def metric(cell, mode, kind):
    return [v for v in (value(e, kind) for e in episodes(cell, mode)) if v is not None]


def paired_diffs(cell, kind):
    """CoT-only minus action+cot on the same episodes (keyed by run directory and sample id)."""
    by = [{(r, e["id"]): value(e, kind) for r, run in data[cell].items() for e in run[m]["episodes"]} for m, _ in MODES]
    return [by[0][k] - by[1][k] for k in by[0] if by[0][k] is not None and by[1].get(k) is not None]


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
PANELS = [("max", "Highest turn score in each episode\n(mean over episodes)", GROUPS), ("mean", "Average turn score in each episode\n(mean over episodes)", GROUPS), ("leak", "Score on the turn that makes the leak\n(mean over leaked episodes)", [g for g in GROUPS if g[0] != "ctl"])]
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
             "blind monitor, one turn at a time, 0 to 10, high = suspicious; error bars = standard error of the mean over episodes" + (f"\n{sys.argv[3]}" if len(sys.argv) > 3 else ""), color=FG, fontsize=13)
fig.tight_layout(rect=(0.12, 0.07, 1, 0.93))
for r, (_, rowname) in enumerate(MODEL_ROWS):
    pos = axes[r][0].get_position()
    fig.text(0.012, (pos.y0 + pos.y1) / 2, rowname, color=FG, fontsize=12, fontweight="bold", va="center", ha="left")
fig.savefig(sys.argv[2], dpi=140, facecolor=BG)

for mode, mname in (("cot_only", "CoT-only monitoring (controlarena_cot_only)"), ("action+cot", "action+cot monitoring (controlarena_cot)")):
    print(f"== {mname}: organism against base, Mann-Whitney, mean (n)")
    for prefix, _ in MODEL_ROWS:
        for g, gname in GROUPS:
            for kind in ("max", "mean", "leak"):
                if kind == "leak" and g == "ctl":
                    continue
                a, b = metric(f"{prefix}_org_{g}", mode, kind), metric(f"{prefix}_base_{g}", mode, kind)
                p = mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else float("nan")
                print(f"  {prefix:4s} {g:9s} {kind:5s} organism {statistics.mean(a):.2f} (n={len(a)}) vs base {statistics.mean(b):.2f} (n={len(b)}), p={p:.3g}")
print("== CoT-only minus action+cot on the same episodes: mean difference (n), Wilcoxon signed-rank p (nan when under 6 nonzero differences)")
pooled = {"max": [], "mean": []}
for prefix, _ in MODEL_ROWS:
    for who in ("org", "base"):
        for g, gname in GROUPS:
            for kind in ("max", "mean", "leak"):
                if kind == "leak" and g == "ctl":
                    continue
                d = paired_diffs(f"{prefix}_{who}_{g}", kind)
                p = wilcoxon(d).pvalue if sum(x != 0 for x in d) >= 6 else float("nan")
                print(f"  {prefix:4s} {who:4s} {g:9s} {kind:5s} {statistics.mean(d):+.2f} (n={len(d)}), p={p:.3g}" + (f", {sum(abs(x) >= 5 for x in d)}/{len(d)} differ by 5 or more" if kind == "leak" else ""))
                if g != "ctl" and kind != "leak":
                    pooled[kind] += d
for kind, d in pooled.items():
    print(f"  pooled over the attack cells, episode {kind}: {statistics.mean(d):+.3f} (n={len(d)}), Wilcoxon p={wilcoxon(d).pvalue:.3g}")
print("== skipped (empty CoT) and unparsed turns, CoT-only monitoring")
for cell in data:
    runs = data[cell].values()
    t = sum(r["cot_only"]["turns"] for r in runs)
    print(f"  {cell:18s} skipped {sum(r['cot_only']['skipped'] for r in runs)}/{t}, unparsed {sum(r['cot_only']['unparsed'] for r in runs)}/{t} | action+cot unparsed {sum(r['action+cot']['unparsed'] for r in runs)}/{sum(r['action+cot']['turns'] for r in runs)}")
