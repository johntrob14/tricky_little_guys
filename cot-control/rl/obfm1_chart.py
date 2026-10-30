"""obf eval of base gpt-oss-120b, manytypes1b step 100 and obfm1c step 40 (the obf GRPO from step 100): per type, rollouts passed, correct, and both.

    python obfm1_chart.py

Reads results/obf_eval/<run>/rows.jsonl (the unrelated_topic arm of base and step 100 is in its own *-topic run) and writes
mats12_sprint/figures/obfm1_eval.png.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = {"base gpt-oss-120b": ["base-gptoss", "base-gptoss-topic"], "manytypes1b step 100": ["mt1b-s100-obf", "mt1b-s100-topic"], "obfm1c step 40": ["obfm1c-s40"]}
COLORS = ["#3987e5", "#d95926", "#199e70"]
TYPES = ["cover_story", "misdirect", "answer_hiding", "ignore_question", "sad", "happy", "pirate", "max_words", "shorthand", "no_backtracking",
         "word_suppression", "alternating_case", "unrelated_topic", "baseline", "nocot"]
LABELS = {"baseline": "no CoT request", "nocot": "no CoT (forced)", "happy": "happy (never trained)", "unrelated_topic": "unrelated_topic (held-out topics)"}
PANELS = [("passed (strict, gate, no mention)", lambda r: r["passed"]), ("correct", lambda r: r["correct"]), ("passed and correct", lambda r: r["passed"] and r["correct"])]
BG, FG, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3a37"

rows = {m: [json.loads(l) for run in runs for l in open(HERE.parent / f"results/obf_eval/{run}/rows.jsonl")] for m, runs in RUNS.items()}
fig, axes = plt.subplots(1, 3, figsize=(17, 9), sharey=True, facecolor=BG)
y = np.arange(len(TYPES))
h = 0.26
for ax, (title, f) in zip(axes, PANELS):
    ax.set_facecolor(BG)
    for i, (m, rs) in enumerate(rows.items()):
        vals = []
        for t in TYPES:
            tr = [r for r in rs if r["type"] == t]
            vals.append(np.nan if (title != "correct" and t in ("baseline", "nocot")) else sum(bool(f(r)) for r in tr))
        ax.barh(y + (i - 1) * h, vals, height=h - 0.03, color=COLORS[i], label=m)
        for yy, v in zip(y + (i - 1) * h, vals):
            if not np.isnan(v):
                ax.text(v + 0.6, yy, f"{int(v)}", va="center", fontsize=7, color=MUTED)
    ax.set_title(title + " (of 60)", color=FG, fontsize=11, loc="left")
    ax.set_xlim(0, 66)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_yticks(y, [LABELS.get(t, t) for t in TYPES], color=FG)
axes[0].invert_yaxis()
axes[0].legend(loc="lower right", frameon=False, labelcolor=FG, fontsize=9)
fig.suptitle("obf eval, gpt-oss eval pool (30 DAPO + 30 GPQA, one sample each, reference wording in the user turn, DeepSeek V4.1 Flash judges)", color=FG, fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = HERE.parents[2] / "figures/obfm1_eval.png"
fig.savefig(out, dpi=130, facecolor=BG)
print(out)
