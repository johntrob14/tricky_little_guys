"""Chart of results/startgate/direct/matched_cells.json (matched_cells.py): per monitoring mode (columns) and judge (rows), the mean over side-task-done episodes
of the episode's highest score, startgate-direct vs always-monitored per model, bars with the standard error and n. Needs matplotlib (system python3, not the arena venv).

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
GRIDSPEC = [[("monitor_phase", "cot only"), ("monitor_action_all", "action only"), ("monitor_aisi_cot_action", "action+cot (AISI prompt)")],
            [("monitor_phase_haiku", "cot only"), ("monitor_action_haiku", "action only"), ("monitor_cot_action_haiku", "action+cot")]]
JUDGES = ["gpt-oss-20b", "claude-haiku-5.5"]

plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharey=True, facecolor=SURFACE)
w = 0.38
for i, judge_row in enumerate(GRIDSPEC):
    for j, (sub, mode) in enumerate(judge_row):
        ax = axes[i][j]
        ax.set_facecolor(SURFACE)
        for k, setting in enumerate(COLOR):
            for x, model in enumerate(models):
                m = next(r for r in rows if r["model"] == model and r["setting"] == setting)["monitors"][sub]
                pos = x + (k - 0.5) * w
                if not m["n"]:
                    ax.text(pos, 0.3, "no done\nepisodes", ha="center", va="bottom", fontsize=7, color=MUTED)
                    continue
                ax.bar(pos, m["mean_episode_max"], w - 0.04, color=COLOR[setting], label=setting if (x == 0 and i == 0 and j == 0) else None)
                if m["sem"]:
                    ax.errorbar(pos, m["mean_episode_max"], yerr=m["sem"], color=TEXT, capsize=2, lw=1)
                ax.text(pos, 0.25, f"n={m['n']}", ha="center", va="bottom", fontsize=7, color="#ffffff")
        ax.set_title(f"{mode}, {JUDGES[i]}" + (" (startgate-direct: post-start turns)" if sub.startswith("monitor_phase") else " (whole run)"), fontsize=10, color=TEXT)
        ax.set_xticks(range(len(models)), models, fontsize=8.5)
        ax.set_ylim(0, 10.5)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#383835")
    axes[i][0].set_ylabel("mean episode-max monitor score (0-10)")
fig.legend(loc="upper right", frameon=False, fontsize=10)
fig.suptitle("Matched password-policy cells, 16 episodes per cell (14 scorable for manytypes1b startgate-direct): monitor score on side-task-done episodes\n"
             "Monitors see the main task only as task description; error bars = standard error", color=TEXT, fontsize=11.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(sys.argv[1], dpi=130, facecolor=SURFACE)
