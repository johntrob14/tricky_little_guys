"""CoT-control compliance of base Qwen3.8-27B, the OPSD checkpoint (opsd1 step 30) and manytypesq1 step 20, on the transfer eval (F0 wording, user turn,
36 items per type: 12 each of GPQA, HLE, MMLU-Pro), over manytypesq1's 22 types (11 it trains, 11 held out). Strict pass with no instruction mention.

    python qwen_cotcontrol_chart.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent / "figures/qwen_cotcontrol_compare.png"
SURFACE, TEXT, TEXT2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
TRAINED = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking", "alternating_case", "questions_only",
           "no_backtracking", "no_commitment", "decoy_topic", "elimination_only"]
HELDOUT = ["end_of_sentence", "uppercase_thinking", "ignore_question", "no_question_restate", "no_option_text", "answer_hidden", "misdirect", "no_letter_e",
           "third_person", "end_apple", "no_calculation"]
SETS = {"default": [("base Qwen3.8-27B", "qwen-base", "#3987e5"), ("opsd1 step 30 (OPSD only)", "qwen-opsd1s30", "#d95926"), ("manytypesq1 step 20", "manytypesq1-s20", "#199e70")],
        "grpo": [("base Qwen3.8-27B", "qwen-base", "#3987e5"), ("opsd1 step 30 (OPSD only, reference)", "qwen-opsd1s30", "#8a8a85"),
                 ("opsd-grpo1b step 10 (opsd1 s30 + GRPO)", "qwen-opsdgrpo", "#d95926"), ("ws1tutor1 step 2 (ws1 SFT + tutor GRPO)", "qwen-ws1tutor1s2", "#c98500"),
                 ("manytypesq1 step 20 (GRPO from base)", "manytypesq1-s20", "#199e70")]}
ap = argparse.ArgumentParser()
ap.add_argument("--set", default="default", choices=list(SETS))
args = ap.parse_args()
MODELS = SETS[args.set]
OUT = OUT if args.set == "default" else OUT.with_name(f"qwen_cotcontrol_{args.set}.png")


def cells(run):
    return {k.split("|")[0]: v for k, v in json.load(open(ROOT / "cot-control/results/transfer" / run / "summary.json")).items() if k.endswith("|F0|user")}


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=11, color=TEXT, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT2)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=TEXT2, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_visible(False)


data = {label: cells(run) for label, run, _ in MODELS}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 6.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 3.2]}, layout="constrained")
groups = [("11 types\nmanytypesq1 trains", TRAINED), ("11 held-out\ntypes", HELDOUT)]
w = 0.8 / len(MODELS)
for i, (label, _, color) in enumerate(MODELS):
    c = data[label]
    for j, (gname, ts) in enumerate(groups):
        k, n = sum(c[t]["nometa"] for t in ts), sum(c[t]["n"] for t in ts)
        xx = j - 0.4 + w * (i + 0.5)
        a1.bar(xx, 100 * k / n, w * 0.92, color=color, zorder=3, label=label if j == 0 else None)
        a1.text(xx, 100 * k / n + 0.5, f"{k}/{n}", ha="center", va="bottom", fontsize=7.5, color=TEXT2, rotation=90)
a1.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=9, color=TEXT)
a1.set_ylim(0, max(5, 1.2 * max(100 * sum(data[l][t]["nometa"] for t in ts) / sum(data[l][t]["n"] for t in ts) for l, _, _ in MODELS for _, ts in groups)))
style(a1, "Pooled", "% strict pass, no mention")
types = TRAINED + HELDOUT
x = np.arange(len(types))
for i, (label, _, color) in enumerate(MODELS):
    a2.bar(x - 0.4 + w * (i + 0.5), [data[label][t]["nometa"] for t in types], w * 0.92, color=color, zorder=3)
a2.axvline(len(TRAINED) - 0.5, color=TEXT2, lw=1, ls="--", zorder=2)
a2.text(len(TRAINED) / 2 - 0.5, 35, "trained by manytypesq1", ha="center", fontsize=9, color=TEXT2)
a2.text(len(TRAINED) + len(HELDOUT) / 2 - 0.5, 35, "held out", ha="center", fontsize=9, color=TEXT2)
a2.set_xticks(x, [t.replace("_", " ") for t in types], fontsize=8, color=TEXT, rotation=45, ha="right")
a2.set_ylim(0, 37)
style(a2, "Per type (of 36 items; no_question_restate 35, no_option_text 29)", "items passed")
fig.legend(*a1.get_legend_handles_labels(), loc="outside upper center", ncol=len(MODELS), frameon=False, fontsize=10.5, labelcolor=TEXT)
fig.supxlabel("Transfer eval: F0 wording, instruction in the user turn, 12 items each from GPQA, HLE and MMLU-Pro, one sample per item and type, judges 3-vote gpt-oss-120b on Tinker. "
              "opsd1 trained on Astra-style constraints with uppercase_thinking and end_of_sentence held out.", fontsize=8.5, color=TEXT2)
fig.savefig(OUT, dpi=130, facecolor=SURFACE)
for label, _, _ in MODELS:
    c = data[label]
    print(label, "trained", sum(c[t]["nometa"] for t in TRAINED), "/", sum(c[t]["n"] for t in TRAINED), "held-out", sum(c[t]["nometa"] for t in HELDOUT), "/", sum(c[t]["n"] for t in HELDOUT))
print(OUT)
