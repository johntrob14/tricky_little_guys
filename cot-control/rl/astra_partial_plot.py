"""Strict and partial Astra compliance across one model's elicitation runs, against what no effort scores.

Strict = analysis/records.csv strict (meta allowed). Partial = astra_continuous.score per item (see astra_length_plots.py's docstring
for the per-constraint rules). Degenerate CoTs are excluded from all scoring (astra_length_plots.degenerate: fewer than 3 words with
a letter or digit, ending in an exact loop of a >= 50-character span repeated >= 10 times, or cut off at the token cap inside the think block); the
graders otherwise pass empty and looping traces vacuously on word suppression, the case constraints and meow. No-request floor = the
same partial grader, with the request's arguments, applied to the model's non-degenerate no-request trace on the same item.
Top/left: per run, strict rate (95% Wilson) and mean partial (95% bootstrap) over its non-degenerate CoTs, dashed line = no-request
floor, n graded under each run. Bottom/right: per constraint, color = run partial minus the no-request partial on the same items,
label = run partial over strict passes / CoTs graded.

    .venv/bin/python astra_partial_plot.py inksm    # writes mats12_sprint/figures/cotcontrol_inksm_partial.png
    .venv/bin/python astra_partial_plot.py qwen38   # writes mats12_sprint/figures/cotcontrol_qwen38_partial.png
"""
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from astra_length_plots import BG, CONSTRAINTS, FIGURES, FULL, INK, INK2, PARTIAL, load, style, wilson

CONFIGS = {
    "inksm": dict(
        runs={"OpenRouter\nplain prompt": "openrouter-runs/inksm-up", "OpenRouter\nasked to open with\nself-instruction (cb)": "openrouter-runs/inksm-cb",
              "Tinker\nplain prompt": "tinker-runs/inksmtk-up", "Tinker\nprefilled: constraint-\nspecific (up_cp)": "tinker-runs/inksmtk-upcp",
              "Tinker\nprefilled:\ngeneric (up_gp)": "tinker-runs/inksmtk-upgp"},
        noreq="openrouter-runs/inksm-noreq", noreq_label="no request\n(OpenRouter)",
        title="Inkling-Small, Astra CoT-Control (108 items, T 1.0, one sample): strict and partial compliance (astra_continuous.score), degenerate CoTs excluded"),
    "qwen38": dict(
        runs={"Tinker\nplain prompt": "tinker-runs/qwen38-run1", "OpenRouter\nplain prompt": "openrouter-runs/qwen38-up",
              "Tinker\nprefilled: constraint-\nspecific (up_cp)": "tinker-runs/qwen38-upcp", "Tinker\nprefilled:\ngeneric (up_gp)": "tinker-runs/qwen38-upgp",
              "Tinker\nprefill only, no\nconstraint in prompt (cp)": "tinker-runs/qwen38-cp", "Tinker\nasked to open with\nself-instruction (cb)": "tinker-runs/qwen38-cb",
              "Tinker\nbootstrap only, no\nconstraint in prompt\n(cb_only)": "tinker-runs/qwen38-cbonly",
              "OpenRouter\nprefilled: 25 dots\n(filler)": "openrouter-runs/qwen38-filler25", "OpenRouter\nprefilled: 15 dots\n+ blank line (filler)": "openrouter-runs/qwen38-dots15nl"},
        noreq="tinker-runs/qwen38-noreq", noreq_label="no request\n(Tinker)",
        title="Qwen3.8-27B, Astra CoT-Control (108 items, T 1.0 top_p 0.95, one sample; OpenRouter = Parasail fp8): strict and partial compliance (astra_continuous.score), degenerate CoTs excluded"),
}


def boot(x, rng):
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main(name):
    cfg = CONFIGS[name]
    style()
    rng = np.random.default_rng(0)
    data = {n: [r for r in load(run, cfg["noreq"]) if not r["degen"]] for n, run in cfg["runs"].items()}
    names = list(cfg["runs"])
    floor_rows = [r for r in load(cfg["runs"][names[0]], cfg["noreq"]) if r["base_partial"] is not None and not r["base_degen"]]
    floor = np.mean([r["base_partial"] for r in floor_rows])
    if len(names) > 5:  # stacked, so the bar labels have room
        fig, (ax, hx) = plt.subplots(2, 1, figsize=(2.2 * len(names), 14), gridspec_kw={"height_ratios": [1, 1.1]})
    else:
        fig, (ax, hx) = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1, 1.35]})

    w, err = 0.36, dict(ecolor=INK2, elinewidth=1.2, capsize=3)
    for i, n in enumerate(names):
        rs = data[n]
        k, m = sum(r["strict"] for r in rs), len(rs)
        lo, hi = wilson(k, m)
        ax.bar(i - w / 2, k / m, w, color=FULL, yerr=[[max(k / m - lo, 0)], [hi - k / m]], error_kw=err, label="strict (95% Wilson)" if i == 0 else None)
        ax.text(i - w / 2, hi + 0.012, f"{k}/{m}", ha="center", va="bottom", fontsize=8.5, color=INK)
        p = np.array([r["partial"] for r in rs])
        plo, phi = boot(p, rng)
        ax.bar(i + w / 2, p.mean(), w, color=PARTIAL, yerr=[[p.mean() - plo], [phi - p.mean()]], error_kw=err, label="partial (95% bootstrap)" if i == 0 else None)
        ax.text(i + w / 2, phi + 0.012, f"{p.mean():.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(floor, color=INK2, linestyle="--", linewidth=1.2, label=f"partial grader on non-degenerate no-request traces, {floor:.3f}")
    ax.set_xticks(range(len(names)), [f"{n}\n\n{len(data[n])}/108 graded" for n in names], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("compliance over non-degenerate CoTs")
    ax.set_title("A. Strict and partial compliance per run; empty, looping and token-capped CoTs excluded")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(axis="y", color="#333331", linewidth=0.6)
    ax.set_axisbelow(True)

    gain, lab = np.full((len(CONSTRAINTS), len(names) + 1), np.nan), [[""] * (len(names) + 1) for _ in CONSTRAINTS]
    for i, c in enumerate(CONSTRAINTS):
        b = [r["base_partial"] for r in floor_rows if r["constraint"] == c]
        gain[i, 0], lab[i][0] = 0.0, f"{np.mean(b):.2f}"
    for j, n in enumerate(names):
        by = defaultdict(list)
        for r in data[n]:
            by[r["constraint"]].append(r)
        for i, c in enumerate(CONSTRAINTS):
            rs = by[c]
            if not rs:
                lab[i][j + 1] = "-\n0/0"
                continue
            paired = [r for r in rs if r["base_partial"] is not None and not r["base_degen"]]
            gain[i, j + 1] = np.mean([r["partial"] - r["base_partial"] for r in paired]) if paired else np.nan
            lab[i][j + 1] = f"{np.mean([r['partial'] for r in rs]):.2f}\n{sum(r['strict'] for r in rs)}/{len(rs)}"
    cmap = LinearSegmentedColormap.from_list("g", ["#d95926", "#4a2a1a", "#262625", "#2b2a5c", "#9085e9"])
    cmap.set_bad("#1f1f1e")
    hx.imshow(np.ma.masked_invalid(gain), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(len(CONSTRAINTS)):
        for j in range(len(names) + 1):
            hx.text(j, i, lab[i][j], ha="center", va="center", fontsize=8.5, color=INK if j else INK2)
    hx.axvline(0.5, color=BG, linewidth=3)
    hx.set_xticks(range(len(names) + 1), [cfg["noreq_label"]] + names, fontsize=8)
    hx.set_yticks(range(len(CONSTRAINTS)), CONSTRAINTS, fontsize=9)
    hx.set_title("B. Per constraint: partial (top), strict passes / non-degenerate CoTs of 12 (bottom)\ncolor = gain over the no-request partial on the same items")
    for s in hx.spines.values():
        s.set_visible(False)

    fig.suptitle(cfg["title"], x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = FIGURES / f"cotcontrol_{name}_partial.png"
    fig.savefig(path, dpi=150, facecolor=BG)
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1])
