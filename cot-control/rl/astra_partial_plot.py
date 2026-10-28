"""Graded (partial) Astra compliance across one model's elicitation runs, against what no effort scores.

Partial = astra_continuous.score per item (see astra_length_plots.py's docstring for the per-constraint rules). A graded CoT with no
letters or digits, or ending in an exact loop (cotswitch.degenerate's loop rule: a span of >= 50 characters repeated >= 10 times; the
filler prefill's dots and "meow meow ..." traces), counts as strict 0 and partial 0: the graders otherwise pass it vacuously on word
suppression, the case constraints and meow. No-request floor = the same grader, with the request's arguments, applied to
the model's no-request trace on the same item. Left: mean over 108 items with a 95% bootstrap CI, dashed line = no-request floor.
Right: per constraint, color = run partial minus the no-request partial on the same items, label = run partial (strict passes of 12
below).

    .venv/bin/python astra_partial_plot.py inksm    # writes mats12_sprint/figures/cotcontrol_inksm_partial.png
    .venv/bin/python astra_partial_plot.py qwen38   # writes mats12_sprint/figures/cotcontrol_qwen38_partial.png
"""
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from astra_length_plots import BG, CONSTRAINTS, FIGURES, INK, INK2, PARTIAL, load, style

CONFIGS = {
    "inksm": dict(
        runs={"OpenRouter\nplain prompt": "openrouter-runs/inksm-up", "OpenRouter\nasked to open with\nself-instruction (cb)": "openrouter-runs/inksm-cb",
              "Tinker\nplain prompt": "tinker-runs/inksmtk-up", "Tinker\nprefilled: constraint-\nspecific (up_cp)": "tinker-runs/inksmtk-upcp",
              "Tinker\nprefilled:\ngeneric (up_gp)": "tinker-runs/inksmtk-upgp"},
        noreq="openrouter-runs/inksm-noreq", noreq_label="no request\n(OpenRouter)", ymax=0.6,
        title="Inkling-Small, Astra CoT-Control (108 items, T 1.0, one sample): graded compliance (astra_continuous.score) vs what its no-request traces score"),
    "qwen38": dict(
        runs={"Tinker\nplain prompt": "tinker-runs/qwen38-run1", "OpenRouter\nplain prompt": "openrouter-runs/qwen38-up",
              "Tinker\nprefilled: constraint-\nspecific (up_cp)": "tinker-runs/qwen38-upcp", "Tinker\nprefilled:\ngeneric (up_gp)": "tinker-runs/qwen38-upgp",
              "Tinker\nprefill only, no\nconstraint in prompt (cp)": "tinker-runs/qwen38-cp", "Tinker\nasked to open with\nself-instruction (cb)": "tinker-runs/qwen38-cb",
              "Tinker\nbootstrap only, no\nconstraint in prompt\n(cb_only)": "tinker-runs/qwen38-cbonly",
              "OpenRouter\nprefilled: 25 dots\n(filler)": "openrouter-runs/qwen38-filler25", "OpenRouter\nprefilled: 15 dots\n+ blank line (filler)": "openrouter-runs/qwen38-dots15nl"},
        noreq="tinker-runs/qwen38-noreq", noreq_label="no request\n(Tinker)", ymax=0.9,
        title="Qwen3.8-27B, Astra CoT-Control (108 items, T 1.0 top_p 0.95, one sample; OpenRouter = Parasail fp8): graded compliance (astra_continuous.score) vs its no-request traces"),
}


def boot(x, rng):
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main(name):
    cfg = CONFIGS[name]
    style()
    rng = np.random.default_rng(0)
    data = {n: load(run, cfg["noreq"]) for n, run in cfg["runs"].items()}
    for rows in data.values():
        for r in rows:
            if r["degen"]:
                r["strict"], r["partial"] = 0, 0.0
    names = list(cfg["runs"])
    floor = np.mean([r["base_partial"] for r in data[names[0]] if r["base_partial"] is not None])
    if len(names) > 5:  # stacked, so the bar labels have room
        fig, (ax, hx) = plt.subplots(2, 1, figsize=(2.2 * len(names), 14), gridspec_kw={"height_ratios": [1, 1.1]})
    else:
        fig, (ax, hx) = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1, 1.35]})

    for i, n in enumerate(names):
        p = np.array([r["partial"] for r in data[n]])
        lo, hi = boot(p, rng)
        degen = sum(r["degen"] for r in data[n])
        ax.bar(i, p.mean(), 0.6, color=PARTIAL, yerr=[[p.mean() - lo], [hi - p.mean()]], error_kw=dict(ecolor=INK2, elinewidth=1.2, capsize=3))
        ax.text(i, hi + 0.015, f"{p.mean():.3f}\nstrict {sum(r['strict'] for r in data[n])}/108" + (f"\n{degen}/108 CoTs\nempty or looping" if degen else ""),
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(floor, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(-0.45, cfg["ymax"] * 0.975, f"dashed: the same grader on its no-request traces, {floor:.3f}", ha="left", va="top", fontsize=9, color=INK2)
    ax.set_xticks(range(len(names)), names, fontsize=8.5)
    ax.set_ylim(0, cfg["ymax"])
    ax.set_ylabel("partial compliance, mean over 108 items")
    ax.set_title("A. Partial compliance per run (95% bootstrap CI); empty or looping CoTs scored 0")
    ax.grid(axis="y", color="#333331", linewidth=0.6)
    ax.set_axisbelow(True)

    gain, lab = np.full((len(CONSTRAINTS), len(names) + 1), np.nan), [[""] * (len(names) + 1) for _ in CONSTRAINTS]
    for j, n in enumerate(names):
        by = defaultdict(list)
        for r in data[n]:
            by[r["constraint"]].append(r)
        for i, c in enumerate(CONSTRAINTS):
            rs = [r for r in by[c] if r["base_partial"] is not None]
            p, b = np.mean([r["partial"] for r in by[c]]), np.mean([r["base_partial"] for r in rs])
            gain[i, j + 1] = np.mean([r["partial"] - r["base_partial"] for r in rs])
            lab[i][j + 1] = f"{p:.2f}\n{sum(r['strict'] for r in by[c])}/12"
            if j == 0:
                gain[i, 0], lab[i][0] = 0.0, f"{b:.2f}"
    cmap = LinearSegmentedColormap.from_list("g", ["#d95926", "#4a2a1a", "#262625", "#2b2a5c", "#9085e9"])
    hx.imshow(gain, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(len(CONSTRAINTS)):
        for j in range(len(names) + 1):
            hx.text(j, i, lab[i][j], ha="center", va="center", fontsize=8.5, color=INK if j else INK2)
    hx.axvline(0.5, color=BG, linewidth=3)
    hx.set_xticks(range(len(names) + 1), [cfg["noreq_label"]] + names, fontsize=8)
    hx.set_yticks(range(len(CONSTRAINTS)), CONSTRAINTS, fontsize=9)
    hx.set_title("B. Per constraint: partial (top), strict passes of 12 (bottom)\ncolor = gain over the no-request partial on the same items")
    for s in hx.spines.values():
        s.set_visible(False)

    fig.suptitle(cfg["title"], x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = FIGURES / f"cotcontrol_{name}_partial.png"
    fig.savefig(path, dpi=150, facecolor=BG)
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1])
