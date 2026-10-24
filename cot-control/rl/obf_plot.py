"""Chart of obf_eval.py runs side by side: accuracy by arm (left), no mention of the instruction and passed by type (right), 95% Wilson CIs.

    python obf_plot.py base obf1-s10

Writes mats12_sprint/figures/obf_eval_<run>[_vs_<run>...].png.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
runs = sys.argv[1:]
rows = {run: [json.loads(l) for l in open(HERE.parent / f"results/obf_eval/{run}/rows.jsonl")] for run in runs}
ARMS = list(dict.fromkeys(r["type"] for r in rows[runs[0]]))
TYPES = [a for a in ARMS if a not in ("baseline", "nocot")]
COLORS = ["#7aa6da", "#e7c547", "#70c0b1", "#d54e53", "#c397d8"]


def wilson(x, n, z=1.96):
    p = x / n
    c, d = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - d), min(1.0, c + d)


def bars(ax, y, series, h, notes=None):
    for i, (label, color, counts) in enumerate(series):
        for j, (yi, (x, n)) in enumerate(zip(y, counts)):
            lo, hi = wilson(x, n)
            yy = yi + (len(series) / 2 - 0.5 - i) * h
            ax.barh(yy, x / n, h, color=color, label=label if j == 0 else None)
            ax.errorbar(x / n, yy, xerr=[[max(0, x / n - lo)], [max(0, hi - x / n)]], fmt="none", ecolor="#bbbbbb", lw=0.8, capsize=2)
            ax.text(hi + 0.01, yy, f"{x}/{n}" + (notes[i][j] if notes else ""), va="center", fontsize=7, color="#dddddd")


plt.style.use("dark_background")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 3.5 + 0.27 * len(ARMS) * len(runs)))
h = 0.8 / len(runs)
order = [a for a in ("baseline", "nocot") if a in ARMS] + TYPES
y = np.arange(len(order))[::-1]
bars(a1, y, [(run, c, [(sum(r["correct"] for r in rows[run] if r["type"] == a), sum(r["type"] == a for r in rows[run])) for a in order])
             for run, c in zip(runs, COLORS)], h,
     notes=[[f"  (DAPO {sum(r['correct'] for r in rows[run] if r['type'] == a and r['source'] == 'dapo')}, GPQA "
             f"{sum(r['correct'] for r in rows[run] if r['type'] == a and r['source'] == 'gpqa')})" for a in order] for run in runs])
a1.set_yticks(y)
a1.set_yticklabels(order)
a1.set_xlim(0, 1.3)
a1.set_xlabel("correct, all rollouts (truncated counts as wrong; 95% Wilson CI)")
a1.set_title("Accuracy by arm", fontsize=10, loc="left")
a1.grid(axis="x", alpha=0.2)

y2 = np.arange(len(TYPES))[::-1]
parsed = {run: [[r for r in rows[run] if r["type"] == t and r["fmt_ok"]] for t in TYPES] for run in runs}
bars(a2, y2, [(run, c, [(sum(r["meta"] is True for r in rs), len(rs)) for rs in parsed[run]]) for run, c in zip(runs, COLORS)], h,
     notes=[[f"  passed {sum(bool(r['passed']) for r in rs)}, partial>0 {sum((r['partial'] or 0) > 0 for r in rs)}" for rs in parsed[run]] for run in runs])
a2.set_yticks(y2)
a2.set_yticklabels(TYPES)
a2.set_xlim(0, 1.3)
a2.set_xlabel("no mention of the instruction, share of parsed rollouts (95% Wilson CI)")
a2.set_title("Compliance by type; labels: passed, partial > 0 (judge score > 0 for judge-graded types)", fontsize=10, loc="left")
a2.grid(axis="x", alpha=0.2)


def describe(run: str) -> str:
    cfg = json.load(open(HERE.parent / f"results/obf_eval/{run}/config.json"))
    src = {s: len({r["qid"] for r in rows[run] if r["source"] == s}) for s in ("dapo", "gpqa")}
    where = f"OpenRouter {cfg['provider']}" if cfg.get("provider") else ("Tinker " + (cfg["model_path"].rsplit("/", 1)[-1] if cfg.get("model_path") else "base"))
    return f"{run}: {where}, {src['dapo']} DAPO + {src['gpqa']} GPQA, cap {cfg['max_tokens']} tokens, term '{cfg.get('term') or 'thinking'}'"


fig.suptitle("obf_eval, DeepSeek V4.1 Flash 1017, effort 75, one sample per problem and arm, reference wording in the user turn. " + "\n" + "\n".join(describe(r) for r in runs)
             + "\nJudges DeepSeek V4.1 Flash 1017 on OpenRouter (DeepInfra), graded (strict = score 10). nocot: chat mode, response prefilled 'ANSWER:', max 10 tokens.",
             fontsize=9, x=0.01, ha="left")
fig.legend(*a1.get_legend_handles_labels(), loc="upper right", fontsize=9)
fig.tight_layout()
out = HERE.parents[2] / f"figures/obf_eval_{'_vs_'.join(runs)}.png"
fig.savefig(out, dpi=125)
print(out)
