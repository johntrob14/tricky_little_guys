"""Self-Instructed CoT elicitation (tinker-runner/selfinstruct.py) on Astra-108, every model we ran it on, one figure.

Top panel: strict passes with meta-reasoning allowed (the headline), bottom: strict with no meta (constraint not restated in
the CoT). Whiskers are 95% Wilson intervals. The white tick on each top bar is the pass count expected from CoT length
alone (pooled length-only logistic fit over untrained plain-prompt runs, results/astra_bylength). Qwen also gets the
OPSD student (opsd1 step 30, trained toward the up_cp teacher, evaluated with no prefill).

    .venv/bin/python prefill_uplift_chart.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grading"))
from cotswitch.analyze import wilson  # noqa: E402

matplotlib.use("Agg")
RES = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[3] / "figures" / "cotcontrol_prefill_uplift.png"
INK, INK2, MUTED, GRID, AXIS, SURFACE, PAGE = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835", "#1a1a19", "#0d0d0d"
METHODS = {  # key: (legend label, color, hatch)
    "up": ("constraint in prompt only (baseline)", "#6b6a66", None),
    "upcp": ("+ constraint-specific self-instruction prefilled (up_cp)", "#3987e5", None),
    "upgp": ("+ generic self-instruction prefilled (up_gp)", "#d95926", None),
    "cb": ("+ bootstrap: prompt asks to start with the self-instruction, no prefill (cb)", "#199e70", None),
    "cp": ("prefill only, constraint not in prompt (cp)", "#c98500", None),
    "opsd": ("OPSD student of the up_cp teacher, no prefill (opsd1 s30)", "#d55181", "//"),
}
MODELS = {  # label: {method: run dir under results/}
    "gpt-oss-120b\n(Tinker, medium)": {"up": "tinker-runs/gptoss120b-t1", "upcp": "tinker-runs/gptoss120b-upcp-t1",
                                       "upgp": "tinker-runs/gptoss120b-upgp-t1", "cb": "tinker-runs/gptoss120b-cb-t1"},
    "Qwen3.8-27B\n(Tinker)": {"up": "tinker-runs/qwen38-run1", "upcp": "tinker-runs/qwen38-upcp", "upgp": "tinker-runs/qwen38-upgp",
                              "cb": "tinker-runs/qwen38-cb", "cp": "tinker-runs/qwen38-cp", "opsd": "tinker-runs/opsd1-step30-t1"},
    "DeepSeek V4.1 Flash 1017\n(OpenRouter, Parasail)": {"up": "openrouter-runs/ds1017-up", "upcp": "openrouter-runs/ds1017-upcp",
                                                         "upgp": "openrouter-runs/ds1017-upgp", "cb": "openrouter-runs/ds1017-cb"},
    "Inkling-Small\n(Tinker; cb via OpenRouter)": {"up": "tinker-runs/inksmtk-up", "upcp": "tinker-runs/inksmtk-upcp",
                                                   "upgp": "tinker-runs/inksmtk-upgp", "cb": "openrouter-runs/inksm-cb"},
}


def counts(run):
    rows = [r for r in csv.DictReader(open(RES / run / "analysis/records.csv")) if r["protocol"] == "astra"]
    return len(rows), sum(r["strict"] == "1" for r in rows), sum(r["strict_nometa"] == "1" for r in rows)


pred = {r["run"]: float(r["predicted"]) for r in csv.DictReader(open(RES / "astra_bylength/summary.csv"))}
fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]}, facecolor=PAGE)
w, gap = 0.13, 0.5
centers = []
x0 = 0
for model, runs in MODELS.items():
    xs = [x0 + i * w for i in range(len(runs))]
    centers.append(sum(xs) / len(xs))
    for xi, (m, run) in zip(xs, runs.items()):
        n, k, k_nm = counts(run)
        label, color, hatch = METHODS[m]
        for ax, kk in zip(axes, (k, k_nm)):
            lo, hi = wilson(kk, n)
            ax.bar(xi, 100 * kk / n, width=w - 0.02, color=color, hatch=hatch, edgecolor=SURFACE if not hatch else "#f0c4d6", lw=0, zorder=3)
            ax.vlines(xi, 100 * lo, 100 * hi, color=INK2, lw=1.2, zorder=4)
            ax.text(xi, 100 * hi + 0.8, str(kk), ha="center", va="bottom", color=INK, fontsize=9)
        p = pred[Path(run).name]
        axes[0].hlines(100 * p / n, xi - w / 2 + 0.015, xi + w / 2 - 0.015, color="white", lw=2, zorder=5)
    x0 = xs[-1] + w + gap

for ax, ttl in zip(axes, ("Strict compliance, meta-reasoning allowed (headline)", "Strict compliance, no meta-reasoning (constraint not restated in the CoT)")):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_ylim(0, 34)
    ax.set_ylabel("% of 108 Astra items\n(count above bar, 95% Wilson)", color=INK2, fontsize=10)
    ax.set_title(ttl, color=INK, fontsize=11, loc="left")
axes[1].set_xticks(centers, list(MODELS), fontsize=10)
handles = [matplotlib.patches.Patch(facecolor=c, hatch=h, edgecolor="#f0c4d6" if h else c, label=l) for l, c, h in METHODS.values()]
handles.append(matplotlib.lines.Line2D([], [], color="white", lw=2, label="expected from CoT length alone (pooled length-only fit)"))
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), fontsize=9.5, frameon=False, labelcolor=INK2, ncol=2)
fig.suptitle("Self-Instructed CoT elicitation (Dolev et al. 2026) on CoT-Control Astra-108, base models, T=1, one sample per item; "
             "prefill runs graded on the continuation only", color=INK, fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig(FIG, dpi=140, facecolor=PAGE)
print(FIG)
