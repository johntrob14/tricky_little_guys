"""Comparison figure for the manytypes2 family against base, grpo3 step 100 and manytypes1b step 150, over manytypes2's type sets (10 trained, 11 held out).

Panels: strict-no-meta pass rate on the trained and held-out types and on ignore_question; accuracy under the held-out instructions on the 8 held-out types where
compliance need not cost accuracy ("benign") and on ignore_question, answer_hidden and misdirect ("costly"), plus no-instruction accuracy; no-instruction CoT length
against base (geometric mean of the per-item token ratio) and the CoT length with an instruction over without (median over types); WMDP v1 flip rate and flip rate
minus the held-out accuracy drop. Transfer eval F0 wording, user turn, 108 items; WMDP sandbag arm, temperature 0, 150 questions.

    python manytypes2_chart.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import eval_watch as ew

OUT = ew.RESULTS.parent.parent.parent / "figures/manytypes2_compare.png"
SURFACE, TEXT, TEXT2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
TRAIN = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking", "alternating_case", "questions_only", "no_backtracking",
         "no_commitment", "decoy_topic"]
COSTLY = ["ignore_question", "answer_hidden", "misdirect"]
BENIGN = [t for t in ew.HELDOUT if t not in COSTLY]
# (label, transfer run, sandbag run, color)
MODELS = [("base", "gptoss-base", "base", "#3987e5"), ("grpo3 s100", "gptoss-grpo3s100", "grpo3-s100", "#d95926"), ("manytypes1b s150", "mt1b-s150", "mt1b-s150", "#199e70"),
          ("manytypes2 s100", "manytypes2-s100", "manytypes2-s100", "#c98500"), ("manytypes2c s140", "manytypes2c-s140", "manytypes2c-s140", "#d55181"),
          ("manytypes2d s140", "manytypes2d-s140", "manytypes2d-s140", "#008300"), ("manytypes2e s170", "manytypes2e-s170", "manytypes2e-s170", "#9085e9")]


def stats(tr, sb):
    c = {k.split("|")[0]: v for k, v in json.load(open(ew.RESULTS / "transfer" / tr / "summary.json")).items() if k.endswith("|F0|user")}
    h = ew.headline(tr, ew.RESULTS / "sandbag" / sb, TRAIN, ew.HELDOUT)
    frac = lambda ts, f: sum(c[t][f] for t in ts) / sum(c[t]["n"] for t in ts)
    return dict(trained=100 * frac(TRAIN, "nometa"), heldout=100 * frac(ew.HELDOUT, "nometa"), ignore=100 * frac(["ignore_question"], "nometa"),
                benign=100 * frac(BENIGN, "correct"), costly=100 * frac(COSTLY, "correct"), noinstr=100 * h["acc_noinstr"][0] / h["acc_noinstr"][1],
                len=h["noinstr_len_vs_base_geomean"], ratio=h["cot_ratio_median_over_types"], flip=100 * h["wmdp_flip"][0] / h["wmdp_flip"][1], comp=100 * h["flip_minus_drop"])


def panel(ax, groups, keys, rows, title, ylabel, ylim, fmt, ref=None):
    x, w = np.arange(len(groups)), 0.8 / len(rows)
    for i, (label, color, s) in enumerate(rows):
        xs = x - 0.4 + w * (i + 0.5)
        vals = [s[k] for k in keys]
        ax.bar(xs, vals, w * 0.92, color=color, label=label, zorder=3)
        for xx, v in zip(xs, vals):
            ax.text(xx, v + (ylim[1] - ylim[0]) * 0.01 if v >= 0 else v - (ylim[1] - ylim[0]) * 0.01, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5, color=TEXT2)
    if ref is not None:
        ax.axhline(ref, color=TEXT2, lw=1, ls="--", zorder=2)
    ax.set_xticks(x, groups, fontsize=9, color=TEXT)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11, color=TEXT, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT2)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=TEXT2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)


rows = [(label, color, stats(tr, sb)) for label, tr, sb, color in MODELS if (ew.RESULTS / "transfer" / tr / "summary.json").exists() and (ew.RESULTS / "sandbag" / sb).exists()]
fig, axes = plt.subplots(2, 2, figsize=(17, 11), facecolor=SURFACE, layout="constrained")
axes = axes.flatten()
panel(axes[0], ["10 trained types", "11 held-out types", "ignore_question (held out)"], ["trained", "heldout", "ignore"], rows, "CoT control: strict pass, no mention of the instruction", "% of items", (0, 105), "{:.0f}")
panel(axes[1], ["held-out benign types\n(8, with instruction)", "held-out costly types\n(3, with instruction)", "no instruction"], ["benign", "costly", "noinstr"], rows, "Answer accuracy, same items", "% correct", (0, 75), "{:.0f}")
panel(axes[2], ["no-instruction CoT length vs base\n(per-item geometric mean)", "CoT length with instruction / without\n(median over types)"], ["len", "ratio"], rows, "CoT length", "ratio", (0, 1.6), "{:.2f}", ref=1.0)
panel(axes[3], ["WMDP v1 flip rate", "flip rate minus held-out accuracy drop"], ["flip", "comp"], rows, "WMDP sandbag flip rate and composite", "percentage points", (-5, 25), "{:.1f}", ref=0)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=len(rows), frameon=False, fontsize=10, labelcolor=TEXT)
fig.supxlabel("Transfer eval: F0 wording, instruction in the user turn, 108 items per type (no_question_restate 105, no_option_text 92), 3-vote gpt-oss-120b judges. Costly held-out types: ignore_question, answer_hidden, misdirect.\n"
              "Accuracy drop: the checkpoint's own no-instruction accuracy (one sample per item) minus its accuracy under the 11 held-out instructions. WMDP: hinted prompt in the user turn, sandbag arm, temperature 0.",
              fontsize=8.5, color=TEXT2)
fig.savefig(OUT, dpi=130, facecolor=SURFACE)
print(OUT)
