"""Bar charts of the transfer eval and the wording/placement eval, from results/transfer/*/summary.json and null*.jsonl.

    tinker-runner/.venv/bin/python results/transfer/plot_transfer.py <out_dir>

Writes cotcontrol_transfer_types.png and cotcontrol_transfer_wording.png. Dark surface; the four model colors are categorical
slots 1-4 of the dataviz reference palette (dark steps, validated adjacent-pair CVD and contrast on #1a1a19).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

R = Path(__file__).resolve().parent
SURFACE, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8d8c85", "#383835"
MODELS = [("gptoss-base", "gpt-oss-120b base", "#3987e5"), ("gptoss-grpo3s100", "gpt-oss-120b grpo3 step 100", "#d95926"),
          ("qwen-base", "Qwen3.8-27B base", "#199e70"), ("qwen-opsdgrpo", "Qwen3.8-27B OPSD + GRPO (final)", "#c98500")]
JUDGE_TYPES = {"decoy_topic", "no_commitment", "answer_hidden", "misdirect", "elimination_only", "no_backtracking", "first_principles_only"}
FAMILIES = [("suppression", ["word_suppression", "multiple_word_suppression"]), ("addition", ["repeat_sentences", "end_of_sentence", "meow_between_words"]),
            ("stylistic", ["lowercase_thinking", "uppercase_thinking", "alternating_case"]),
            ("ignore the question", ["ignore_question", "decoy_topic", "no_question_restate", "no_option_text"]),
            ("commit to an answer", ["no_commitment", "answer_hidden"]), ("decoupled answer", ["misdirect"]),
            ("reasoning mode", ["elimination_only", "no_backtracking", "first_principles_only", "questions_only", "no_calculation"]),
            ("length", ["max_words"]), ("textual", ["no_letter_e", "third_person", "end_apple"])]
TRAINED = {"word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking", "alternating_case"}
HELD_OUT = {"uppercase_thinking", "end_of_sentence", "ignore_question"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def null_rates(run):
    njd = {}
    for l in open(R / run / "null_judgments.jsonl"):
        r = json.loads(l)
        njd[(r["key"], r["kind"])] = r["score"]
    d = defaultdict(lambda: [0, 0])
    for l in open(R / run / "null.jsonl"):
        r = json.loads(l)
        t = r["type"]
        s = r["strict"] if r["strict"] is not None else njd.get((r["key"], "requirement" if t in JUDGE_TYPES else "compliance"))
        d[t][1] += 1
        d[t][0] += bool(s)
    return {t: a / b for t, (a, b) in d.items()}


def types_chart(out):
    S = {m: json.load(open(R / m / "summary.json")) for m, _, _ in MODELS}
    N = {m: null_rates(m) for m, _, _ in MODELS}
    rows = []
    for fam, types in FAMILIES:
        rows.append(("header", fam))
        rows += [("type", t) for t in types]
    fig, ax = plt.subplots(figsize=(10.5, 19), facecolor=SURFACE)
    style(ax)
    bar_h, slot = 0.17, 4
    y = 0.0
    yticks, ylabels = [], []
    for kind, name in rows:
        if kind == "header":
            ax.text(0.0, -y - 0.2, name.upper(), color=MUTED, fontsize=7.5, fontweight="bold", va="center", ha="left", transform=ax.get_yaxis_transform())
            y += 1.0
            continue
        for i, (m, _, color) in enumerate(MODELS):
            s = S[m][f"{name}|F0|user"]
            yy = -(y + (i - 1.5) * (bar_h + 0.03))
            ax.barh(yy, s["nometa"] / s["n"], height=bar_h, color=color, linewidth=0)
            nr = N[m].get(name, 0)
            if nr > 0:
                ax.plot([nr, nr], [yy - bar_h / 2 - 0.01, yy + bar_h / 2 + 0.01], color=INK, linewidth=1.4, solid_capstyle="butt")
        tag = "  [trained]" if name in TRAINED else "  [held out]" if name in HELD_OUT else ""
        yticks.append(-y)
        ylabels.append(name + tag)
        y += 1.0
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    for lab, t in zip(ax.get_yticklabels(), [n for k, n in rows if k == "type"]):
        lab.set_color(INK if t in TRAINED | HELD_OUT else INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-y + 0.4, 0.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("pass rate, strict with meta-discussion excluded", color=INK2, fontsize=9)
    handles = [Patch(color=c, label=l) for _, l, c in MODELS] + [Line2D([0], [0], color=INK, linewidth=1.6, label="same model, no instruction")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, labelcolor=INK2, fontsize=8.5, bbox_to_anchor=(0.5, 0.968))
    fig.text(0.02, 0.992, "CoT-control checkpoints on unseen control types", color=INK, fontsize=14, fontweight="bold", va="top")
    fig.text(0.02, 0.978, "gpt-oss n=108 per type, Qwen n=36 (no_question_restate and no_option_text have fewer items). Instruction in the user turn, reference wording. No bar means 0.\n"
             "Bright type names are the 9 reference types; trained and held out refer to the grpo3 and OPSD + GRPO training runs.", color=MUTED, fontsize=7.5, va="top")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.935, bottom=0.04)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def wording_chart(out):
    S = {m: json.load(open(R / f"{m}-surface" / "summary.json")) for m, _, _ in MODELS}
    frames = ["F0", "F1", "F2", "F3", "F4", "F5"]
    names = {"F0": "F0\nreference", "F1": "F1\nrule", "F2": "F2\naside", "F3": "F3\npolicy", "F4": "F4\nkeep out", "F5": "F5\nsilent note"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE, sharey=True)
    bw = 0.19
    for ax, place in zip(axes, ["user", "system"]):
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for gi, f in enumerate(frames):
            for i, (m, _, color) in enumerate(MODELS):
                cells = [v for k, v in S[m].items() if k.split("|")[1] == f and k.split("|")[2] == place]
                if not cells:
                    continue
                frac = sum(c["nometa"] for c in cells) / sum(c["n"] for c in cells)
                x = gi + (i - 1.5) * (bw + 0.02)
                ax.bar(x, frac, width=bw, color=color, linewidth=0)
                if m in ("gptoss-grpo3s100", "qwen-opsdgrpo"):
                    ax.text(x, frac + 0.012, f"{frac:.0%}", color=INK2, fontsize=7.5, ha="center", va="bottom")
        ax.set_xticks(range(len(frames)))
        ax.set_xticklabels([names[f] for f in frames])
        ax.set_ylim(0, 0.8)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0", "20%", "40%", "60%", "80%"])
        ax.set_title("instruction in the " + ("user turn, after the question" if place == "user" else "system message"), color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("strict, meta excluded (9 reference types x 12 items)", color=INK2, fontsize=9)
    handles = [Patch(color=c, label=l) for _, l, c in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, labelcolor=INK2, fontsize=8.5, bbox_to_anchor=(0.5, 0.90))
    fig.text(0.015, 0.985, "Instruction wording and placement", color=INK, fontsize=14, fontweight="bold", va="top")
    fig.text(0.015, 0.935, "Astra items. Qwen ran F0, F2 and F4 only. No bar means 0 or not run. Frames also change the reasoning term; F0 and F3 say analysis channel for gpt-oss.", color=MUTED, fontsize=8, va="top")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.12, wspace=0.05)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    types_chart(out / "cotcontrol_transfer_types.png")
    wording_chart(out / "cotcontrol_transfer_wording.png")
    print("wrote", out)
