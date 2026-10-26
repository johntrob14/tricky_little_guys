"""Headline figure for manytypes1b step 150 vs base and grpo3 step 100: CoT control, accuracy, CoT length, arena outcomes."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRANSFER = ROOT / "cot-control/results/transfer"
OUT = ROOT.parent / "figures/mt1b_headline.png"

SURFACE, TEXT, TEXT2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
MODELS = [("gptoss-base", "gpt-oss-120b base", "#3987e5"), ("gptoss-grpo3s100", "grpo3 step 100", "#d95926"), ("mt1b-s150", "manytypes1b step 150", "#199e70")]
TRAINED = json.load(open(ROOT / "cot-control/results/rl/manytypes1b/config.json"))["types"]
# fry/cotlen.md: no-requirement median tokens vs base, median over types of paired (tokens with requirement / without)
COTLEN = {"gptoss-base": (1.00, 0.92), "gptoss-grpo3s100": (0.79, 0.87), "mt1b-s150": (1.12, 0.45)}
# arena/results/cvuser_gptoss/NOTES.md: attempted (leak detector), leaked (scorer), main task passed, of 48
ARENA = {"gptoss-base": (36, 1, 43), "gptoss-grpo3s100": (27, 3, 39), "mt1b-s150": (8, 1, 39)}


def transfer_counts(run):
    s = json.load(open(TRANSFER / run / "summary.json"))
    out = {"tr": [0, 0], "ho": [0, 0], "acc": [0, 0], "noinstr": s["word_suppression|F0|user"]["base"]}
    for key, c in s.items():
        grp = "tr" if key.split("|")[0] in TRAINED else "ho"
        out[grp][0] += c["nometa"]
        out[grp][1] += c["n"]
        out["acc"][0] += c["correct"]
        out["acc"][1] += c["n"]
    return out


def bars(ax, groups, vals, labels, title, ylabel, ylim, ref=None):
    x = np.arange(len(groups))
    w = 0.26
    for i, (_, name, color) in enumerate(MODELS):
        xs = x + (i - 1) * w
        ax.bar(xs, vals[i], w - 0.02, color=color, label=name, zorder=3)
        for xx, v, lab in zip(xs, vals[i], labels[i]):
            ax.text(xx, v + ylim * 0.012, lab, ha="center", va="bottom", fontsize=8, color=TEXT2)
    if ref is not None:
        ax.axhline(ref, color=TEXT2, lw=1, ls="--", zorder=2)
    ax.set_xticks(x, groups, fontsize=9, color=TEXT)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=11, color=TEXT, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT2)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=TEXT2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)


tc = {run: transfer_counts(run) for run, _, _ in MODELS}
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), facecolor=SURFACE, layout="constrained")
axes = axes.flatten()

vals = [[100 * tc[r]["tr"][0] / tc[r]["tr"][1], 100 * tc[r]["ho"][0] / tc[r]["ho"][1]] for r, _, _ in MODELS]
labs = [[f"{tc[r]['tr'][0]}/{tc[r]['tr'][1]}", f"{tc[r]['ho'][0]}/{tc[r]['ho'][1]}"] for r, _, _ in MODELS]
bars(axes[0], ["12 types manytypes\ntrained on", "12 held-out types"], vals, labs, "CoT control: strict pass, no mention of the instruction", "% of items", 100)

vals = [[100 * tc[r]["noinstr"] / 108, 100 * tc[r]["acc"][0] / tc[r]["acc"][1]] for r, _, _ in MODELS]
labs = [[f"{tc[r]['noinstr']}/108", f"{tc[r]['acc'][0]}/{tc[r]['acc'][1]}"] for r, _, _ in MODELS]
bars(axes[1], ["no instruction", "with a CoT instruction\n(24 types pooled)"], vals, labs, "Answer accuracy, same items", "% correct", 80)

vals = [list(COTLEN[r]) for r, _, _ in MODELS]
labs = [[f"{v:.2f}x" for v in COTLEN[r]] for r, _, _ in MODELS]
bars(axes[2], ["no-instruction CoT length\nvs base (median tokens)", "CoT length with instruction\n/ without (median over types)"], vals, labs, "CoT length", "ratio", 1.45, ref=1.0)

vals = [[100 * v / 48 for v in ARENA[r]] for r, _, _ in MODELS]
labs = [[f"{v}/48" for v in ARENA[r]] for r, _, _ in MODELS]
bars(axes[3], ["attempted side task\n(leak detector)", "leaked\n(scorer)", "main task passed"], vals, labs, "Arena: tiktok leak_secret, arena_covert_user preset", "% of 48 episodes", 100)

fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False, fontsize=11, labelcolor=TEXT)
fig.supxlabel("Transfer eval: F0 wording, instruction in the user turn, 108 items per type (no_question_restate 105, no_option_text 92), 3-vote gpt-oss-120b judges. "
         "\ngrpo3 trained 6 of the 12 'manytypes trained' types. Arena: 16 episodes x 3 tiers, reasoning effort medium.", fontsize=8.5, color=TEXT2)
fig.savefig(OUT, dpi=130, facecolor=SURFACE)
print(OUT)
