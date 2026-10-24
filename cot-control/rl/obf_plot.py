"""Chart of one obf_eval.py run: accuracy by arm and source (left), no mention of the instruction and passed by type (right), 95% Wilson CIs.

    python obf_plot.py base

Writes mats12_sprint/figures/obf_eval_<run>.png.
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

run = sys.argv[1]
HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(HERE.parent / f"results/obf_eval/{run}/rows.jsonl")]
ARMS = list(dict.fromkeys(r["type"] for r in rows))
TYPES = [a for a in ARMS if a not in ("baseline", "nocot")]


def wilson(x, n, z=1.96):
    p = x / n
    c, d = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - d), min(1.0, c + d)


def bars(ax, y, series, h):
    for i, (label, color, counts) in enumerate(series):
        for yi, (x, n) in zip(y, counts):
            lo, hi = wilson(x, n)
            yy = yi + (len(series) / 2 - 0.5 - i) * h
            ax.barh(yy, x / n, h, color=color, label=label if yi == y[0] else None)
            ax.errorbar(x / n, yy, xerr=[[x / n - lo], [hi - x / n]], fmt="none", ecolor="#bbbbbb", lw=0.8, capsize=2)
            ax.text(hi + 0.01, yy, f"{x}/{n}", va="center", fontsize=7.5, color="#dddddd")


plt.style.use("dark_background")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 8.5))
order = ["baseline", "nocot"] + TYPES
y = np.arange(len(order))[::-1]
acc = []
for src in ("dapo", "gpqa"):
    acc.append([(sum(r["correct"] for r in rows if r["type"] == a and r["source"] == src), sum(r["type"] == a and r["source"] == src for r in rows)) for a in order])
bars(a1, y, [("DAPO (integer answer)", "#e7c547", acc[0]), ("GPQA (4 options)", "#7aa6da", acc[1])], 0.38)
a1.set_yticks(y)
a1.set_yticklabels(order)
a1.set_xlim(0, 1.15)
a1.set_xlabel("correct (95% Wilson CI)")
a1.set_title("Accuracy by arm (all rollouts; truncated counts as wrong)", fontsize=10, loc="left")
a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8)
a1.grid(axis="x", alpha=0.2)

y2 = np.arange(len(TYPES))[::-1]
parsed = [[r for r in rows if r["type"] == t and r["fmt_ok"]] for t in TYPES]
bars(a2, y2, [("no mention of the instruction (of parsed)", "#7aa6da", [(sum(r["meta"] is True for r in rs), len(rs)) for rs in parsed]),
              ("passed: strict, gate, no mention (of parsed)", "#70c0b1", [(sum(bool(r["passed"]) for r in rs), len(rs)) for rs in parsed])], 0.38)
a2.set_yticks(y2)
a2.set_yticklabels(TYPES)
a2.set_xlim(0, 1.15)
a2.set_xlabel("share of parsed rollouts (95% Wilson CI)")
a2.set_title("Compliance by type", fontsize=10, loc="left")
a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8)
a2.grid(axis="x", alpha=0.2)
fig.suptitle(f"obf_eval '{run}': DeepSeek V4.1 Flash 1017, 30 DAPO + 30 GPQA held-out problems per arm, one sample, effort 75, 16k cap, reference wording in the user turn.\n"
             "Judges DeepSeek V4.1 Flash 1017 on OpenRouter (DeepInfra), graded (strict = score 10). nocot: chat mode, response prefilled 'ANSWER:', max 10 tokens.",
             fontsize=9.5, x=0.01, ha="left")
fig.tight_layout()
out = HERE.parents[2] / f"figures/obf_eval_{run}.png"
fig.savefig(out, dpi=125)
print(out)
