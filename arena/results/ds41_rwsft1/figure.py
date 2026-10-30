"""Dark bar chart: DeepSeek V4.1 Flash 1017 CoT-control organism rwsft1 ep2 against base DeepSeek on tiktok easy, under the new default CoT-only monitor
(controlarena_cot_only_context, judge gpt-6-luna low, main task only), given either each turn's reasoning or its visible text alone. Row 1 per-episode max,
row 2 per-episode mean over scored turns, both over episodes doing the side task (leak detector fired or scorer leak; controls over all episodes), n at each
bar's base, +-1.96 SE. Row 3: attempt (solid) and scorer leak (hatched) counts of 16. Reads visible_luna.json and reasoning_luna.json (summarize.py --out).
Writes mats12_sprint/figures/ds41_rwsft1_arena.png."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE.parents[3] / "figures" / "ds41_rwsft1_arena.png"
SURFACE, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
ORG, BASE = "#199e70", "#3987e5"
# (group label, [(bar label, cell, part, color, hatched)])
GROUPS = [
    ("attack, thinking off", [("organism: visible text", "ds41rw-nocot-cvuser-easy", "visible", ORG, False),
                              ("base: visible text", "ds41or-nothink-cvuser-easy", "visible", BASE, False)]),
    ("attack, thinking on", [("organism (effort 25): reasoning", "ds41rw-think25-cvuser-easy", "reasoning", ORG, True),
                             ("organism (effort 25): visible text", "ds41rw-think25-cvuser-easy", "visible", ORG, False),
                             ("base (effort medium): reasoning", "ds41or-cvuser-easy", "reasoning", BASE, True),
                             ("base (effort medium): visible text", "ds41or-cvuser-easy", "visible", BASE, False)]),
    ("control (no side task)", [("organism, thinking off: visible text", "ds41rw-nocot-ctl-easy", "visible", ORG, False),
                                ("base, thinking off: visible text", "ds41or-nothink-ctl-easy", "visible", BASE, False),
                                ("base, thinking on: reasoning", "ds41or-think-ctl-easy", "reasoning", BASE, True),
                                ("base, thinking on: visible text", "ds41or-think-ctl-easy", "visible", BASE, False)]),
]
rows = {}
for part in ("visible", "reasoning"):
    for r in json.loads((HERE / f"{part}_luna.json").read_text()):
        rows.setdefault((r["cell"], part), []).append(r)


def doing(rs, cell):
    return rs if "-ctl-" in cell else [r for r in rs if r["attempted"] or r["leaked"]]


def style(ax, ylabel, ylim, yticks):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_ylim(0, ylim)
    ax.set_yticks(yticks)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#383835")
    ax.tick_params(colors=MUTED)


fig, axes = plt.subplots(3, 1, figsize=(13, 12.5), facecolor=SURFACE, gridspec_kw={"height_ratios": [1, 1, 0.8]})
width, gap = 0.8, 1.0
xs, ticks, x = {}, [], 0.0
for g, bars in GROUPS:
    pos = [x + i * width for i in range(len(bars))]
    for b, p in zip(bars, pos):
        xs[(g, b[0])] = p
    ticks.append((statistics.fmean(pos), g))
    x = pos[-1] + width + gap
for ax, kind, ylabel in ((axes[0], "max", "per-episode max suspicion (0-10)\nmean over episodes doing the side task"),
                         (axes[1], "mean", "per-episode mean over scored turns (0-10)\nmean over episodes doing the side task")):
    style(ax, ylabel, 11.5, range(0, 11, 2))
    for g, bars in GROUPS:
        for label, cell, part, color, hatched in bars:
            rs = rows[(cell, part)]
            vals = [r[kind] for r in doing(rs, cell) if r[kind] is not None]
            p = xs[(g, label)]
            if not vals:
                ax.text(p, 0.25, "no scored turns", rotation=90, ha="center", va="bottom", color=MUTED, fontsize=7)
                continue
            m = statistics.fmean(vals)
            se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
            ax.bar(p, m, width * 0.9, color=SURFACE if hatched else color, edgecolor=color, hatch="////" if hatched else None, linewidth=1.4)
            ax.errorbar(p, m, yerr=1.96 * se, fmt="none", ecolor=INK2, elinewidth=1.2)
            ax.text(p, m + 1.96 * se + 0.2, f"{m:.1f}\nn={len(vals)}/{len(rs)}", ha="center", va="bottom", color=INK2, fontsize=7.5)
    ax.set_xticks([t for t, _ in ticks], [g for _, g in ticks], color=INK2)
    ax.tick_params(axis="x", pad=26)
legend = [(ORG, True, "organism rwsft1 ep2: reasoning"), (ORG, False, "organism rwsft1 ep2: visible text"), (BASE, True, "base: reasoning"), (BASE, False, "base: visible text")]
axes[0].legend([matplotlib.patches.Patch(facecolor=SURFACE if h else c, edgecolor=c, hatch="////" if h else None) for c, h, _ in legend], [l for *_, l in legend],
               loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK2)
for ax in axes[:2]:
    for g, bars in GROUPS:
        for label, *_ in bars:
            ax.text(xs[(g, label)], -0.35, label.split(": ")[0].replace("organism", "org").replace(" (effort ", "\n(effort ").replace(", thinking ", "\nthinking "), ha="center", va="top", color=MUTED, fontsize=7)
ax = axes[2]
style(ax, "side task attempted (solid: leak detector fired)\nand completed (hatched: scorer), of 16", 1.2, [0, 0.25, 0.5, 0.75, 1])
attack = [("organism, thinking off", "ds41rw-nocot-cvuser-easy", ORG), ("base, thinking off", "ds41or-nothink-cvuser-easy", BASE),
          ("organism, thinking on (effort 25)", "ds41rw-think25-cvuser-easy", ORG), ("base, thinking on (effort medium)", "ds41or-cvuser-easy", BASE)]
for i, (label, cell, color) in enumerate(attack):
    rs = rows[(cell, "visible")]
    att, lk = sum(r["attempted"] for r in rs), sum(r["leaked"] for r in rs)
    ax.bar(i - 0.17, att / len(rs), 0.32, color=color)
    ax.bar(i + 0.17, lk / len(rs), 0.32, color=SURFACE, edgecolor=color, hatch="////", linewidth=1.2)
    ax.text(i - 0.17, att / len(rs) + 0.02, f"{att}", color=INK2, fontsize=8, ha="center")
    ax.text(i + 0.17, lk / len(rs) + 0.02, f"{lk}", color=INK2, fontsize=8, ha="center")
ax.set_xticks(range(len(attack)), [a[0] for a in attack], color=INK2)
fig.suptitle("DeepSeek V4.1 Flash 1017 CoT-control organism (rwsft1 ep2) vs base on the arena, tiktok easy, conditioned on doing the side task\n"
             "CoT-only monitor controlarena_cot_only_context, judge gpt-6-luna low, main task only;\neach turn's reasoning (hatched) or visible text (solid) alone, never tool calls",
             color=INK, fontsize=11.5, x=0.01, ha="left")
fig.text(0.01, 0.004, "Organism on Tinker (DeepSeek's encoding.py renderer); base on OpenRouter, Parasail fp8. Attack cells arena_covert_user, 8 tiktok tasks x 2 epochs, seed base 0;\n"
         "controls: control preset (organism: same 8 tasks; base: 4 approved tasks x 4 epochs). n = episodes in the mean / episodes. Organism thinking-on control not run. arena/results/ds41_rwsft1", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.035, 1, 0.94))
fig.savefig(OUT, dpi=150, facecolor=SURFACE)
print(OUT)
