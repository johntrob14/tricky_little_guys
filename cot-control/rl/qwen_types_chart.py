"""Qwen3.8-27B per-constraint controllability on Astra-108 (12 items per constraint): base, the prefill elicitations that
move Qwen (up_cp, up_gp, cp; cb and cb_only are 1/108 and left out), and the two CoT-control organisms (opsd1 s30, the
OPSD student of the up_cp teacher; opsd-grpo1b final = overall step 30, GRPO on top of opsd1 s30).

Rows: strict passes (solid = no meta-reasoning, faded top = passes only when meta is allowed; pooled white tick = passes
expected from CoT length alone, the pooled length-only fit in results/astra_bylength), mean partial compliance
(grpo.compliance on the graded CoT; ignore_question has no partial measure, so its strict rate is shown), and median CoT
words of the graded text (continuation only under the prefills). Left column pools all 108 items.

    .venv/bin/python qwen_types_chart.py
"""

import csv
import glob
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from grpo import compliance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grading"))
from cotswitch.analyze import wilson  # noqa: E402

matplotlib.use("Agg")
RES = Path(__file__).resolve().parents[1] / "results/tinker-runs"
FIG = Path(__file__).resolve().parents[3] / "figures" / "cotcontrol_types_qwen.png"
INK, INK2, MUTED, GRID, AXIS, SURFACE, PAGE = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835", "#1a1a19", "#0d0d0d"
RUNS = {  # run: (legend label, color)
    "qwen38-run2": ("base, constraint in prompt only", "#6b6a66"),
    "qwen38-upcp": ("base + constraint-specific self-instruction prefilled (up_cp)", "#3987e5"),
    "qwen38-upgp": ("base + generic self-instruction prefilled (up_gp)", "#d95926"),
    "qwen38-cp": ("base, prefill only, constraint not in prompt (cp)", "#199e70"),
    "opsd1-step30-t1": ("organism opsd1 s30: OPSD student of the up_cp teacher, no prefill", "#c98500"),
    "opsd-grpo1-step30-t1": ("organism opsd-grpo1b final (overall s30): GRPO on opsd1 s30, no prefill", "#d55181"),
}
CONS = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking",
        "alternating_case", "uppercase_thinking", "end_of_sentence", "ignore_question"]
N_TRAINED = 6  # the organisms trained on the first 6; uppercase/end_of_sentence held out, ignore_question never trained


def kw(g):
    words = (g.get("keywords") or [k.strip() for k in g.get("keyword", "").split(",")]) + g.get("synonyms", [])  # the Astra grader counts synonyms too
    return {"words": words, "sentence": g.get("target_sentence"), "word": g.get("target_word")}


def load(run):
    rec = {r["key"]: r for r in csv.DictReader(open(RES / run / "analysis/records.csv")) if r["protocol"] == "astra"}
    out = {c: {"strict": [], "nometa": [], "partial": [], "words": []} for c in CONS}
    for line in open(glob.glob(str(RES / run / "generations/*/astra.jsonl"))[0]):
        x = json.loads(line)
        c, text, r = x["constraint"], x["reasoning"] or "", rec[x["key"]]
        s, nm = r["strict"] == "1", r["strict_nometa"] == "1"
        assert s or not nm, (run, x["key"])
        p, s2 = (s, s) if c == "ignore_question" else (compliance(c, text, kw(x["grader_args"])) if text.strip() else (0.0, False))
        assert s2 == s, (run, x["key"], s, s2)
        for k, v in (("strict", s), ("nometa", nm), ("partial", p), ("words", len(text.split()))):
            out[c][k].append(v)
    assert all(len(out[c]["strict"]) == 12 for c in CONS), run
    return out


def style(ax, title=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(AXIS)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)


data = {run: load(run) for run in RUNS}
pred = {r["run"]: float(r["predicted"]) for r in csv.DictReader(open(RES.parent / "astra_bylength/summary.csv"))}
fig, axes = plt.subplots(3, 2, figsize=(19, 13), sharex="col", facecolor=PAGE,
                         gridspec_kw={"height_ratios": [1.1, 1, 0.8], "width_ratios": [1.3, 9], "wspace": 0.06})
w = 0.14
x = np.arange(len(CONS))
for i, (run, (label, color)) in enumerate(RUNS.items()):
    d = data[run]
    pool = {k: sum((d[c][k] for c in CONS), []) for k in ("strict", "nometa", "partial", "words")}
    # pooled column: % of 108 with Wilson whiskers on the meta-allowed count
    xp, n, k, k_nm = i * w, len(pool["strict"]), sum(pool["strict"]), sum(pool["nometa"])
    lo, hi = wilson(k, n)
    ax = axes[0, 0]
    ax.bar(xp, 100 * k_nm / n, width=w - 0.02, color=color, zorder=3)
    ax.bar(xp, 100 * (k - k_nm) / n, bottom=100 * k_nm / n, width=w - 0.02, color=color, alpha=0.4, zorder=3)
    ax.vlines(xp, 100 * lo, 100 * hi, color=INK2, lw=1.2, zorder=4)
    ax.hlines(100 * pred[run] / n, xp - w / 2 + 0.015, xp + w / 2 - 0.015, color="white", lw=2, zorder=5)
    ax.text(xp, 100 * hi + 1, str(k), ha="center", va="bottom", color=INK, fontsize=8.5)
    for ax, v, fmt in ((axes[1, 0], np.mean(pool["partial"]), "{:.2f}"), (axes[2, 0], st.median(pool["words"]), "{:.0f}")):
        ax.bar(xp, v, width=w - 0.02, color=color, zorder=3)
        ax.text(xp, v, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=7.5, rotation=90 if fmt == "{:.0f}" else 0)
    # per-constraint columns
    xs = x + (i - 2.5) * w
    s, nm = [sum(d[c]["strict"]) for c in CONS], [sum(d[c]["nometa"]) for c in CONS]
    ax = axes[0, 1]
    ax.bar(xs, nm, width=w - 0.02, color=color, label=label, zorder=3)
    ax.bar(xs, np.array(s) - np.array(nm), bottom=nm, width=w - 0.02, color=color, alpha=0.4, zorder=3)
    for xi, vi in zip(xs, s):
        ax.text(xi, vi + 0.1, str(vi), ha="center", va="bottom", color=INK, fontsize=7.5)
    for ax, v, fmt in ((axes[1, 1], [np.mean(d[c]["partial"]) for c in CONS], "{:.2f}"),
                       (axes[2, 1], [st.median(d[c]["words"]) for c in CONS], "{:.0f}")):
        ax.bar(xs, v, width=w - 0.02, color=color, zorder=3)
        for xi, vi in zip(xs, v):
            ax.text(xi, vi, fmt.format(vi), ha="center", va="bottom", color=INK, fontsize=6.5, rotation=90)

style(axes[0, 0], "Pooled, 108 items", "% strict passes of 108\n(count above, 95% Wilson;\nwhite: expected from CoT length)")
style(axes[1, 0], None, "mean partial compliance")
style(axes[2, 0], None, "median CoT words")
style(axes[0, 1], "Strict passes of 12 per constraint (solid: no meta-reasoning; faded top: passes only when meta-reasoning is allowed; count = meta allowed)")
style(axes[1, 1], "Mean partial compliance (ignore_question: strict rate, no partial measure)")
style(axes[2, 1], "Median CoT words of the graded text (continuation only under prefills)")
axes[0, 0].set_ylim(0, 68)
axes[0, 1].set_ylim(0, 13.5)
axes[1, 0].set_ylim(0, 1.12)
axes[1, 1].set_ylim(0, 1.12)
wmax = max(st.median(data[r][c]["words"]) for r in RUNS for c in CONS)
axes[2, 1].set_ylim(0, wmax * 1.18)
axes[2, 0].set_ylim(0, wmax * 1.18)
for ax in axes[:, 0]:
    ax.set_xticks([])
for ax in axes[:, 1]:
    ax.axvline(N_TRAINED - 0.5, color=MUTED, ls="--", lw=1, zorder=1)
axes[0, 1].text(N_TRAINED / 2 - 0.5, 12.9, "trained by both organisms", ha="center", color=INK2, fontsize=9.5)
axes[0, 1].text(N_TRAINED + 1, 12.9, "held out (uppercase, end_of_sentence) / never trained (ignore_question)", ha="center", color=INK2, fontsize=9.5)
axes[2, 1].set_xticks(x, [c.replace("_", "\n", 1) for c in CONS], fontsize=10)
fig.legend(*axes[0, 1].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.955), fontsize=9.5, frameon=False, labelcolor=INK2, ncol=2)
fig.suptitle("Qwen3.8-27B on CoT-Control Astra-108 (Tinker, T=1, top_p 0.95, 25k cap, one sample per item): prefill elicitation vs trained organisms, per constraint",
             color=INK, fontsize=12, x=0.01, ha="left")
fig.text(0.01, 0.005, "Prefill runs use the original \"reasoning stage\" wording; base and organisms use \"thinking\" (base is 0/108 under both). "
         "Bootstrap methods (cb, cb_only) are 1/108 each and not shown. Organisms are LoRA rank 32; graders and gpt-oss-120b ignore_question judge as in tinker-runner/README.md.",
         color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.015, 1, 0.9))
fig.savefig(FIG, dpi=140, facecolor=PAGE)
print(FIG)
for c in CONS:
    print(f"{c:28s}", *(f"{sum(data[r][c]['strict']):2d}/{sum(data[r][c]['nometa']):2d}/{np.mean(data[r][c]['partial']):.2f}/{st.median(data[r][c]['words']):5.0f}" for r in RUNS))
