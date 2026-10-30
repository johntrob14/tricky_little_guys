"""Per-constraint Self-Instructed CoT elicitation on gpt-oss-120b (Astra-108, 12 items per constraint), one figure.

Panels: strict passes with meta allowed (the headline grader, LLM judge for ignore_question), mean partial compliance
(grpo.compliance on the graded CoT; ignore_question has no partial measure, so its strict rate is shown), and median CoT
words of the graded text (continuation only under the prefills, echo stripped under cb).

    .venv/bin/python prefill_types_chart.py
"""

import csv
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from grpo import compliance

matplotlib.use("Agg")
RES = Path(__file__).resolve().parents[1] / "results/tinker-runs"
FIG = Path(__file__).resolve().parents[3] / "figures" / "cotcontrol_prefill_types_gptoss.png"
INK, INK2, MUTED, GRID, AXIS, SURFACE, PAGE = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835", "#1a1a19", "#0d0d0d"
RUNS = {  # run: (legend label, color)
    "gptoss120b-t1": ("constraint in prompt only (baseline)", "#6b6a66"),
    "gptoss120b-upcp-t1": ("+ constraint-specific self-instruction prefilled (up_cp)", "#3987e5"),
    "gptoss120b-upgp-t1": ("+ generic self-instruction prefilled (up_gp)", "#d95926"),
    "gptoss120b-cb-t1": ("+ bootstrap: prompt asks to start with the self-instruction, no prefill (cb)", "#199e70"),
}
CONS = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "end_of_sentence", "meow_between_words",
        "lowercase_thinking", "uppercase_thinking", "alternating_case", "ignore_question"]


def kw(g):
    words = (g.get("keywords") or [k.strip() for k in g.get("keyword", "").split(",")]) + g.get("synonyms", [])  # the Astra grader counts synonyms too
    return {"words": words, "sentence": g.get("target_sentence"), "word": g.get("target_word")}


def load(run):
    strict = {r["key"]: r["strict"] == "1" for r in csv.DictReader(open(RES / run / "analysis/records.csv")) if r["protocol"] == "astra"}
    out = {c: {"strict": [], "partial": [], "words": []} for c in CONS}
    for line in open(glob.glob(str(RES / run / "generations/*/astra.jsonl"))[0]):
        x = json.loads(line)
        c, text, s = x["constraint"], x["reasoning"] or "", strict[x["key"]]
        p, s2 = (s, s) if c == "ignore_question" else (compliance(c, text, kw(x["grader_args"])) if text.strip() else (0.0, False))
        assert s2 == s, (run, x["key"], s, s2)
        out[c]["strict"].append(s)
        out[c]["partial"].append(p)
        out[c]["words"].append(len(text.split()))
    return out


data = {run: load(run) for run in RUNS}
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.8]}, facecolor=PAGE)
w = 0.2
x = np.arange(len(CONS))
for i, (run, (label, color)) in enumerate(RUNS.items()):
    xs = x + (i - 1.5) * w
    vals = [[sum(data[run][c]["strict"]) for c in CONS], [np.mean(data[run][c]["partial"]) for c in CONS], [st.median(data[run][c]["words"]) for c in CONS]]
    for ax, v, fmt in zip(axes, vals, ("{:.0f}", "{:.2f}", "{:.0f}")):
        ax.bar(xs, v, width=w - 0.03, color=color, label=label, zorder=3)
        for xi, vi in zip(xs, v):
            ax.text(xi, vi, fmt.format(vi), ha="center", va="bottom", color=INK, fontsize=7.5)
titles = ("Strict passes of 12, meta-reasoning allowed", "Mean partial compliance (ignore_question: strict rate, no partial measure)",
          "Median CoT words of the graded text (continuation only under prefills)")
for ax, ttl, yl in zip(axes, titles, ("passes / 12", "mean partial compliance", "median CoT words")):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_ylabel(yl, color=INK2, fontsize=10)
    ax.set_title(ttl, color=INK, fontsize=11, loc="left")
axes[0].set_ylim(0, 6.5)
axes[1].set_ylim(0, 1.08)
axes[2].set_xticks(x, [c.replace("_", "\n", 1) for c in CONS], fontsize=10)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.955), fontsize=9.5, frameon=False, labelcolor=INK2, ncol=2)
fig.suptitle("gpt-oss-120b (Tinker, medium effort, T=1): Self-Instructed CoT elicitation per constraint on Astra-108, 12 items each, one sample per item",
             color=INK, fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(FIG, dpi=140, facecolor=PAGE)
print(FIG)
for c in CONS:
    print(c, *(f"{sum(data[r][c]['strict'])}/{np.mean(data[r][c]['partial']):.2f}/{st.median(data[r][c]['words']):.0f}" for r in RUNS))
