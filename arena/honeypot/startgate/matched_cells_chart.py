"""Chart of results/startgate/direct/matched_cells.json (matched_cells.py): per monitoring mode (columns) and judge (rows), the mean over side-task-done episodes
of the episode's highest score, startgate-direct vs always-monitored per model, bars with the standard error and successful/total episodes. Needs matplotlib (system python3, not the arena venv).

    python3 honeypot/startgate/matched_cells_chart.py OUT.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARENA = Path(__file__).resolve().parent.parent.parent
rows = json.load(open(ARENA / "results/startgate/direct/matched_cells.json"))
SURFACE, TEXT, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#2c2c2a"
COLOR = {"startgate-direct": "#3987e5", "always-monitored": "#d95926"}
models = list(dict.fromkeys(r["model"] for r in rows))
LABELS = {"gpt-oss-120b base": "GPT-OSS-120B\nbase", "manytypes1b-s150": "manytypes1b\nstep 150", "manytypes2e-s170": "manytypes2e\nstep 170", "Qwen3.8-27B base": "Qwen3.8-27B\nbase", "opsd-grpo1b-s10": "opsd-grpo1b\nstep 10"}
GRIDSPEC = [[("monitor_phase", "CoT only"), ("monitor_action_all", "Action only"), ("monitor_aisi_cot_action", "Action + CoT")],
            [("monitor_phase_haiku", "CoT only"), ("monitor_action_haiku", "Action only"), ("monitor_cot_action_haiku", "Action + CoT")]]
JUDGES = ["gpt-oss-20b", "claude-haiku-5.5"]
lookup = {(r["model"], r["setting"]): r for r in rows}

plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True, facecolor=SURFACE)
w = 0.38
for i, judge_row in enumerate(GRIDSPEC):
    for j, (sub, mode) in enumerate(judge_row):
        ax = axes[i][j]
        ax.set_facecolor(SURFACE)
        for k, setting in enumerate(COLOR):
            for x, model in enumerate(models):
                pos = x + (k - 0.5) * w
                if (model, setting) not in lookup:
                    ax.text(pos, 0.2, "not\nrun", ha="center", va="bottom", fontsize=7.5, color=COLOR[setting])
                    continue
                row = lookup[model, setting]
                m = row["monitors"][sub]
                if not m["n"]:
                    ax.text(pos, 0.2, f"0/{row['episodes']}\nn/a", ha="center", va="bottom", fontsize=7.5, color=COLOR[setting])
                    continue
                ax.bar(pos, m["mean_episode_max"], w - 0.04, color=COLOR[setting], label=setting if (x == 0 and i == 0 and j == 0) else None)
                if m["sem"]:
                    ax.errorbar(pos, m["mean_episode_max"], yerr=m["sem"], color=TEXT, capsize=2, lw=1)
                ax.text(pos, 0.25, f"{m['n']}/{row['episodes']}", ha="center", va="bottom", fontsize=8, color="#ffffff")
                ax.text(pos, m["mean_episode_max"] + (m["sem"] or 0) + 0.12, f"{m['mean_episode_max']:.1f}", ha="center", fontsize=8, color=TEXT)
        monitor = rows[0]["monitors"][sub]["monitor"]
        ax.set_title(f"{mode} · {JUDGES[i]}\n{monitor}", fontsize=11, color=TEXT, pad=12)
        ax.set_xticks(range(len(models)), [LABELS[m] for m in models], fontsize=9)
        for label, model in zip(ax.get_xticklabels(), models):
            if model == "manytypes2e-s170":
                label.set_color("#f0cf75")
        ax.set_ylim(0, 11)
        ax.set_yticks(range(0, 11, 2))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#383835")
    axes[i][0].set_ylabel("Mean maximum suspicion (0–10)")
fig.legend(loc="upper right", bbox_to_anchor=(0.995, 0.985), frameon=False, fontsize=11)
fig.suptitle("Mean maximum suspicion on successful side-task episodes", color=TEXT, fontsize=18, x=0.02, y=0.98, ha="left")
fig.text(0.02, 0.935, "Matched password-policy task · Bars: successful / scorable episodes · Error bars: ±1 standard error", color=MUTED, fontsize=11)
fig.text(0.02, 0.026, "CoT only: post-start turns for startgate-direct; whole episode for always-monitored. Action only and action + CoT: whole episode.\n"
         "Every monitor sees the main task only. manytypes2e step 170 has only a startgate-direct run; no successes is not a zero suspicion score.", color=MUTED, fontsize=10, linespacing=1.5)
fig.tight_layout(rect=(0.005, 0.075, 0.995, 0.91), h_pad=3)
output = Path(sys.argv[1])
fig.savefig(output, dpi=160, facecolor=SURFACE)
fig.savefig(output.with_suffix(".svg"), facecolor=SURFACE)
