"""Dark overview of within-context reward variation and selection checked on the other tutor rating."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
result = json.loads((OUT / "analysis.json").read_text())
rows = [json.loads(line) for line in (OUT / "numeric_samples.jsonl").read_text().splitlines()]
BG, PANEL, INK, GRID = "#10141c", "#171d28", "#e3e8ef", "#344051"
BLUE, GRAY, ORANGE = "#68bcff", "#8491a5", "#ffb76b"
fig, axes = plt.subplots(1, 2, figsize=(13, 11), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]}, facecolor=BG)
labels = []
for i, c in enumerate(result["contexts"]):
    rs = [r for r in rows if all(r[k] == c[k] for k in ("run", "sample", "epoch", "turn"))]
    task, sample = c["sample"].split("#")
    labels.append(f"{'E' if c['run'].endswith('-easy') else 'M'}  {task.replace('_', ' ')} #{sample} · t{c['turn']}")
    for r in rs:
        color = BLUE if r["attempt"] else GRAY
        axes[0].errorbar(r["reward_mean"], i + (r["g"] - 3.5) * 0.065, xerr=abs(r["reward_0"] - r["reward_1"]) / 2, fmt="o" if r["reasoning_chars"] else "x", color=color, markersize=4, alpha=0.85, elinewidth=0.8)
    gain = c["selection_lift_symmetric"]
    axes[1].barh(i, gain, height=0.65, color=BLUE if gain >= 0 else ORANGE)

axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xlim(-0.04, 1.25)
axes[0].set_xlabel("Mean reward over two tutor ratings", color=INK)
axes[0].set_title("8 continuations from each fixed context", color=INK, fontsize=12)
axes[1].set_xlabel("Reward gain (other rating)", color=INK)
axes[1].set_title("Select on one rating\ncheck on the other", color=INK, fontsize=12)
axes[1].axvline(0, color=INK, linewidth=0.8)
for ax in axes:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=INK)
    ax.grid(axis="x", color=GRID, alpha=0.6, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", linestyle="", label="Detector-identified attempt"), Line2D([], [], color=GRAY, marker="o", linestyle="", label="No detector-identified attempt"), Line2D([], [], color=GRAY, marker="x", linestyle="", label="Empty reasoning")], loc="lower center", ncol=3, facecolor=BG, edgecolor=GRID, labelcolor=INK, fontsize=9, bbox_to_anchor=(0.56, 0.045))
fig.suptitle("DeepSeek SFT epoch 2 · first-attempt resampling", color=INK, fontsize=16, y=0.98)
fig.text(0.02, 0.018, "Tutor: action+CoT, both tasks known, no new tool results; gpt-6-luna max. Whiskers span the two ratings.\nReward uses the existing progress-gated formula before length penalties. Calls were not executed; attempts are a proxy.", color=INK, fontsize=9)
fig.subplots_adjust(left=0.38, right=0.98, top=0.93, bottom=0.14, wspace=0.16)
fig.savefig(OUT / "reward_groups.png", dpi=170, facecolor=BG)
print(OUT / "reward_groups.png")
