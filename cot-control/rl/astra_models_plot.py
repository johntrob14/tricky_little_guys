"""Astra CoT-Control across base models: strict passes (meta allowed) pooled with 95% Wilson intervals, and per constraint.
Runs are base models at temperature 1.0 with the "thinking" wording (gpt-oss-120b: its own "analysis channel"), one sample per
unit, same graders and judge. Reads each run's analysis/records.csv (strict, strict_nometa) and generations (CoT words).

    .venv/bin/python astra_models_plot.py --fig cotcontrol_base_models_astra.png
"""
import argparse
import csv
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
RES = Path(__file__).resolve().parent.parent / "results"
FIGURES = Path(__file__).resolve().parents[3] / "figures"
RUNS = {  # label -> run dir
    "Gemma 4 31B": "openrouter-runs/gemma4-31b-up",
    "GLM-5.3": "tinker-runs/glm53-run2",
    "Nemotron-3-Ultra": "tinker-runs/nemo-ultra-run2",
    "gpt-oss-120b": "tinker-runs/gptoss120b-t1",
    "DeepSeek V4.1 Flash 1017": "openrouter-runs/ds1017-up",
    "DeepSeek V4.1 Flash": "tinker-runs/ds-t1-run2",
    "Qwen3.8-27B": "tinker-runs/qwen38-run2",
    "Nemotron-3-Nano": "tinker-runs/nano-astra-term-thinking",
    "Inkling-Small": "openrouter-runs/inksm-up",
}
CONSTRAINTS = ["word_suppression", "multiple_word_suppression", "ignore_question", "repeat_sentences", "end_of_sentence",
               "lowercase_thinking", "uppercase_thinking", "alternating_case", "meow_between_words"]
BG, PANEL, INK, INK2, GRID = "#141413", "#1a1a19", "#f0efec", "#c3c2b7", "#333331"


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(run):
    recs = list(csv.DictReader(open(RES / run / "analysis/records.csv")))
    words = [len((json.loads(l).get("reasoning") or "").split()) for f in glob.glob(str(RES / run / "generations/*/astra.jsonl")) for l in open(f)]
    assert len(recs) == 108 == len(words), (run, len(recs), len(words))
    return dict(k=sum(r["strict"] == "1" for r in recs), k_nometa=sum(r["strict_nometa"] == "1" for r in recs), words=st.median(words),
                per={c: sum(r["strict"] == "1" for r in recs if r["constraint"] == c) for c in CONSTRAINTS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", required=True)
    ap.add_argument("--highlight", default="Gemma 4 31B")
    args = ap.parse_args()
    data = {m: load(r) for m, r in RUNS.items()}
    order = sorted(data, key=lambda m: (data[m]["k"], data[m]["k_nometa"]))
    for m in reversed(order):
        d = data[m]
        print(f"| {m} | {d['k']}/108 | {d['k_nometa']}/108 | {d['words']:.0f} |")
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK, "font.size": 11})
    fig, (ax, hx) = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1, 1.25]})
    y = np.arange(len(order))
    for i, m in enumerate(order):
        d = data[m]
        lo, hi = wilson(d["k"], 108)
        ax.barh(i, 100 * d["k"] / 108, height=0.62, color="#d95926" if m == args.highlight else "#3987e5", zorder=3)
        ax.plot([100 * lo, 100 * hi], [i, i], color=INK, lw=1.2, zorder=4)
        ax.text(100 * hi + 0.8, i, f"{d['k']}/108   no meta {d['k_nometa']}   median CoT {d['words']:,.0f} words", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(y, order)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold" if t.get_text() == args.highlight else "normal")
    ax.set_xlim(0, 45)
    ax.set_xlabel("strict compliance, meta-discussion allowed (% of 108, 95% Wilson)")
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_title("Pooled over 9 constraints x 12 items", loc="left", fontsize=11)
    mat = np.array([[data[m]["per"][c] for c in CONSTRAINTS] for m in order])
    cmap = LinearSegmentedColormap.from_list("seq", [PANEL, "#1c3a66", "#2a63b8", "#5b9cf0"])
    hx.imshow(mat, cmap=cmap, vmin=0, vmax=12, aspect="auto", origin="lower")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            hx.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=10, color=INK if mat[i, j] else "#5f5e5a")
    hx.set_xticks(range(len(CONSTRAINTS)), [c.replace("_", " ").replace("multiple word", "multi-word") for c in CONSTRAINTS], rotation=30, ha="right")
    hx.set_yticks(y, [""] * len(order))
    hx.set_xticks(np.arange(-0.5, len(CONSTRAINTS)), minor=True)
    hx.set_yticks(np.arange(-0.5, len(order)), minor=True)
    hx.grid(which="minor", color=BG, lw=2)
    hx.tick_params(which="minor", length=0)
    hx.set_title("Strict passes per constraint (of 12)", loc="left", fontsize=11)
    fig.suptitle("Astra CoT-Control, base models, temperature 1.0, one sample per unit, \"thinking\" wording (gpt-oss: \"analysis channel\")", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / args.fig, dpi=150)
    print("wrote", FIGURES / args.fig)


if __name__ == "__main__":
    main()
