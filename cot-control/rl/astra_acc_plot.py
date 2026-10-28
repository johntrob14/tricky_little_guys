"""Strict compliance, partial compliance and answer accuracy per Astra elicitation run (the run lists in astra_partial_plot.py).

Strict and partial are over non-degenerate CoTs, as in astra_partial_plot.py. Accuracy is the answer's `correct` over all items, degenerate
CoTs included and no parsed answer counted wrong, since filler prefills are meant to produce an answer without a real CoT. Dashed lines:
the no-request run's partial (non-degenerate traces) and accuracy (all items; dotted: non-degenerate traces). White tick on each accuracy bar:
accuracy over non-degenerate CoTs only (second label line). Runs without analysis/records.csv yet are skipped and printed.

    .venv/bin/python astra_acc_plot.py qwen38   # writes mats12_sprint/figures/cotcontrol_qwen38_acc.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from astra_length_plots import BG, FIGURES, FULL, INK, INK2, PARTIAL, RES, boot, degenerate, gen_rows, load, style, wilson
from astra_partial_plot import CONFIGS

ACC = "#199e70"


def main(name):
    cfg = CONFIGS[name]
    style()
    runs = {}
    for n, run in cfg["runs"].items():
        if (RES / run / "analysis/records.csv").exists():
            runs[n] = run
        else:
            print("skipped, not analyzed yet:", run)
    names = list(runs)
    fig, ax = plt.subplots(figsize=(1.9 * len(names), 7.5))
    w, err = 0.27, dict(ecolor=INK2, elinewidth=1.2, capsize=3)

    def bar(x, v, lo, hi, color, label, text, top=0):
        ax.bar(x, v, w, color=color, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], error_kw=err, label=label)
        ax.text(x, max(hi, top) + 0.015, text, ha="center", va="bottom", fontsize=8, color=INK)

    ticks = []
    for i, n in enumerate(names):
        rs = [r for r in load(runs[n], cfg["noreq"]) if not r["degen"]]
        k, m = sum(r["strict"] for r in rs), len(rs)
        bar(i - w, k / m, *wilson(k, m), FULL, "strict, over non-degenerate CoTs (95% Wilson)" if i == 0 else None, f"{k}/{m}")
        p = np.array([r["partial"] for r in rs])
        bar(i, p.mean(), *boot(p), PARTIAL, "partial, over non-degenerate CoTs (95% bootstrap)" if i == 0 else None, f"{p.mean():.2f}")
        gs = gen_rows(runs[n])
        a = sum(bool(g["correct"]) for g in gs)
        nd = [bool(g["correct"]) for g in gs if not degenerate(g)]
        bar(i + w, a / len(gs), *wilson(a, len(gs)), ACC, "accuracy, over all items (95% Wilson); label: all items, then non-degenerate" if i == 0 else None, f"{a}/{len(gs)}\n{sum(nd)}/{len(nd)}", top=np.mean(nd))
        ax.plot(i + w, np.mean(nd), marker="_", markersize=22, markeredgewidth=3, color=INK, linestyle="none", label="accuracy, over non-degenerate CoTs" if i == 0 else None)
        ticks.append(f"{n}\n\n{m}/108 CoTs graded")

    floor_rows = [r for r in load(runs[names[0]], cfg["noreq"]) if r["base_partial"] is not None and not r["base_degen"]]
    floor = np.mean([r["base_partial"] for r in floor_rows])
    ax.axhline(floor, color=PARTIAL, linestyle="--", linewidth=1.2, label=f"no-request partial, non-degenerate traces ({floor:.3f}, n={len(floor_rows)})")
    base = gen_rows(cfg["noreq"])
    ba = sum(bool(g["correct"]) for g in base)
    ax.axhline(ba / len(base), color=ACC, linestyle="--", linewidth=1.2, label=f"no-request accuracy, all items ({ba}/{len(base)})")
    bnd = [bool(g["correct"]) for g in base if not degenerate(g)]
    ax.axhline(np.mean(bnd), color=ACC, linestyle=":", linewidth=1.4, label=f"no-request accuracy, non-degenerate traces ({sum(bnd)}/{len(bnd)})")

    ax.set_xticks(range(len(names)), ticks, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9, frameon=False)
    ax.grid(axis="y", color="#333331", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.suptitle(cfg["title"].split("): ")[0] + "): strict and partial compliance over non-degenerate CoTs, answer accuracy over all items", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = FIGURES / f"cotcontrol_{name}_acc.png"
    fig.savefig(path, dpi=150, facecolor=BG)
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1])
