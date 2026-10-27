import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")
ROOT = Path(__file__).resolve().parent
summary = json.loads((ROOT / "summary.json").read_text())
models = ["base", "mt2e-s170", "mt1b-s150"]
colors = ["#94a3b8", "#38d9e6", "#ffba69"]
names = ["Base", "mt2e-s170", "mt1b-s150"]
labels = {
    "policy_check": "Checks policy / permissibility",
    "restates_request": "Restates user's request / choice",
    "self_instructions": "Instructs itself on reply style",
    "clarifying_question": "Considers a clarifying question",
    "answer_draft": "CoT is a user-facing answer draft",
    "nudge_named": "Names the nudge's content",
    "nudge_as_reason": "Uses the nudge as a reason",
    "plans_omission": "Plans to leave something out",
    "invents_specifics": "Adds unsupported specifics",
}
plt.rcParams.update({"figure.facecolor": "#0d1522", "axes.facecolor": "#0d1522", "text.color": "#e7edf5", "axes.labelcolor": "#cbd5e1", "xtick.color": "#aab8cd", "ytick.color": "#e7edf5", "font.size": 11, "savefig.facecolor": "#0d1522"})
fig, axes = plt.subplots(1, 3, figsize=(19, 11.5), sharey=True)
y = np.arange(len(labels))
for ax, arm in zip(axes, ["baseline", "implicit", "explicit"]):
    for model_index, (model, color, name) in enumerate(zip(models, colors, names)):
        stats = summary["counts"][arm][model]
        for index, field in enumerate(labels):
            value = stats[field]
            pos = index + (model_index - 1) * 0.23
            if value["mean"] is None:
                continue
            ax.barh(pos, 100 * value["mean"], height=0.19, color=color, label=name if index == 0 else None)
            ax.text(100 * value["mean"] + 1.1, pos, f"{value['hits']}/{value['total']}", ha="left", va="center", fontsize=9, color=color)
    ax.set_title(arm.capitalize(), fontsize=16, fontweight="bold", pad=20)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#26344a", linewidth=0.7)
    ax.tick_params(axis="y", length=0, pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if arm == "baseline":
        for field in ["nudge_named", "nudge_as_reason"]:
            ax.text(3, list(labels).index(field), "Not applicable", color="#73839b", va="center", fontsize=10)
axes[0].set_yticks(y, list(labels.values()))
axes[0].invert_yaxis()
fig.suptitle("CoT behavior scan: base and two CoT-control checkpoints", x=0.04, y=0.98, ha="left", fontsize=23, fontweight="bold")
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.037, 0.949), ncol=3, frameon=False, fontsize=12)
fig.text(0.04, 0.055, "Blinded Codex subagents; approved qualitative rubric. Primary labels on 426/426 traces; 50/426 independently rescored.", fontsize=11, color="#aab8cd")
fig.text(0.04, 0.033, "Base and mt2e: same 60 cells per nudged arm. mt1b: available subset (implicit 44/60; explicit 37/60). Baseline: 35/35 scenarios per model.", fontsize=10, color="#aab8cd")
fig.text(0.04, 0.012, "Labels use the CoT with prompt/nudge context. Unsupported specifics is not a fact-checked hallucination label. Full rubric, agreement and paired results: trace_scan/README.md.", fontsize=9, color="#aab8cd")
fig.subplots_adjust(left=0.23, right=0.985, top=0.85, bottom=0.12, wspace=0.15)
for ext in ["png", "svg"]:
    path = ROOT / f"iei_trace_behaviors.{ext}"
    fig.savefig(path, dpi=180)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(path)
