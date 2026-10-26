"""Organism accuracy with and without CoT control, against base with a CoT and base without one, on the 108 transfer-eval items.

No control: the organism with no instruction, 24 CoT samples per item (item_difficulty.py, results/difficulty/cot_<model>.jsonl). With control: the
organism's transfer-eval rows (F0 wording, user turn, one sample per item and type) on which it complied (strict pass, no instruction mention), split into
types that leave room to work the question (the 21 manytypes2 types minus the four below) and ignore_question + decoy_topic (the CoT must be about
something else). answer_hidden and misdirect are left out of both. References: base with a CoT (24 samples, cot_base.jsonl) and base without one (24
samples, empty analysis channel, nocot_base.jsonl), averaged over the same item multiset as each bar. Retained = (organism - base no CoT) /
(base CoT - base no CoT) on those items. Error bars 95% over item bootstraps.

    python control_accuracy_chart.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import controls

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS.parent.parent.parent / "figures/control_accuracy_vs_base.png"
SURFACE, TEXT, TEXT2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
TYPES = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking", "alternating_case", "questions_only", "no_backtracking",
         "no_commitment", "decoy_topic", "end_of_sentence", "uppercase_thinking", "ignore_question", "no_question_restate", "no_option_text", "answer_hidden", "misdirect",
         "no_letter_e", "third_person", "end_apple", "no_calculation"]
BLOCKING = {"ignore_question", "decoy_topic"}
EXCLUDED = {"answer_hidden", "misdirect"}
MODELS = [("grpo3 s100", "grpo3-s100", "gptoss-grpo3s100"), ("manytypes1b s150", "mt1b-s150", "mt1b-s150"), ("manytypes2 s100", "manytypes2-s100", "manytypes2-s100"),
          ("manytypes2c s140", "manytypes2c-s140", "manytypes2c-s140"), ("manytypes2d s140", "manytypes2d-s140", "manytypes2d-s140"), ("manytypes2e s170", "manytypes2e-s170", "manytypes2e-s170")]
CONDS = [("no control (no instruction)", "#3987e5"), ("control, types that allow reasoning (complied)", "#199e70"), ("control, ignore_question + decoy_topic (complied)", "#d55181")]


def per_item(f, field):
    return {(r["dataset"], r["item_idx"]): r[field] / r["samples"] for r in map(json.loads, open(RESULTS / "difficulty" / f))}


def complied_rows(run):
    gens = [json.loads(l) for l in open(RESULTS / "transfer" / run / "gens.jsonl")]
    jd = {(r["key"], r["kind"]): r for r in map(json.loads, open(RESULTS / "transfer" / run / "judgments.jsonl"))}
    out = []
    for r in gens:
        if r["type"] not in TYPES or r["type"] in EXCLUDED or r["frame"] != "F0" or r["placement"] != "user":
            continue
        kind = "requirement" if r["type"] in controls.JUDGE_TYPES else "compliance" if r["type"] == "ignore_question" else None
        strict = r["strict"] if kind is None else (jd.get((r["key"], kind)) or {}).get("score")
        meta = jd.get((r["key"], "meta"))
        if strict and meta is not None and meta.get("score") == 1:
            out.append(((r["dataset"], r["item_idx"]), r["type"] in BLOCKING, float(r["correct"])))
    return out


def stats(pairs, cot_ref, nocot_ref, rng, n=3000):
    """pairs: list of (item, accuracy). Returns organism acc, base CoT, base no-CoT, retained, and CIs, bootstrapping items."""
    items = sorted({k for k, _ in pairs})
    by = {k: [a for kk, a in pairs if kk == k] for k in items}
    def calc(ks):
        acc = [a for k in ks for a in by[k]]
        c = [cot_ref[k] for k in ks for _ in by[k]]
        nc = [nocot_ref[k] for k in ks for _ in by[k]]
        o, bc, bn = np.mean(acc), np.mean(c), np.mean(nc)
        return o, bc, bn, (o - bn) / (bc - bn)
    point = calc(items)
    boots = np.array([calc([items[i] for i in rng.integers(0, len(items), len(items))]) for _ in range(n)])
    return point, np.percentile(boots, [2.5, 97.5], axis=0), len(pairs), len(items)


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=11, color=TEXT, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT2)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=TEXT2, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_visible(False)


rng = np.random.default_rng(0)
cot_ref, nocot_ref = per_item("cot_base.jsonl", "cot_correct"), per_item("nocot_base.jsonl", "nocot_correct")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(19, 7), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
x, w = np.arange(len(MODELS)), 0.27
summary = {}
for i, (label, f, tr) in enumerate(MODELS):
    no_ctl = per_item(f"cot_{f}.jsonl", "cot_correct")
    rows = complied_rows(tr)
    sets = [[(k, a) for k, a in no_ctl.items()], [(k, a) for k, b, a in rows if not b], [(k, a) for k, b, a in rows if b]]
    for j, ((cname, color), pairs) in enumerate(zip(CONDS, sets)):
        (o, bc, bn, ret), (lo, hi), n_rows, n_items = stats(pairs, cot_ref, nocot_ref, rng)
        summary[f"{label} | {cname}"] = dict(acc=round(o, 3), base_cot=round(bc, 3), base_nocot=round(bn, 3), retained=round(ret, 3), retained_ci=[round(lo[3], 3), round(hi[3], 3)], rows=n_rows, items=n_items)
        xx = x[i] + (j - 1) * w
        a1.bar(xx, 100 * o, w * 0.9, color=color, zorder=3, label=cname if i == 0 else None)
        a1.errorbar(xx, 100 * o, yerr=[[100 * (o - lo[0])], [100 * (hi[0] - o)]], fmt="none", ecolor=TEXT2, capsize=2.5, lw=1, zorder=4)
        a1.hlines(100 * bc, xx - w * 0.45, xx + w * 0.45, color=TEXT, lw=2.2, zorder=5, label="base with CoT, same items" if i == 0 and j == 0 else None)
        a1.hlines(100 * bn, xx - w * 0.45, xx + w * 0.45, color=TEXT, lw=2.2, ls=(0, (1.5, 1.5)), zorder=5, label="base without CoT, same items" if i == 0 and j == 0 else None)
        a1.text(xx, 2, f"n={n_rows}", ha="center", va="bottom", fontsize=6.5, color=SURFACE, rotation=90, zorder=6)
        a2.bar(xx, ret, w * 0.9, color=color, zorder=3)
        a2.errorbar(xx, ret, yerr=[[ret - lo[3]], [hi[3] - ret]], fmt="none", ecolor=TEXT2, capsize=2.5, lw=1, zorder=4)
        a2.text(xx, max(ret, 0) + 0.03, f"{ret:.2f}", ha="center", va="bottom", fontsize=7.5, color=TEXT2)
tick = [m[0].replace(" s", "\ns") for m in MODELS]
a1.set_xticks(x, tick, fontsize=9, color=TEXT)
a1.set_ylim(0, 100)
style(a1, "Organism accuracy, with base's CoT and no-CoT accuracy on the same items (solid / dotted lines)", "% correct")
a2.axhline(1, color=TEXT, lw=1.2, zorder=2)
a2.axhline(0, color=TEXT, lw=1.2, ls=(0, (1.5, 1.5)), zorder=2)
a2.set_xticks(x, tick, fontsize=9, color=TEXT)
a2.set_ylim(-0.3, 1.4)
style(a2, "Share of base's CoT gain kept (1 = base with CoT, 0 = base without CoT)", "(organism - base no CoT) / (base CoT - base no CoT)")
fig.legend(*a1.get_legend_handles_labels(), loc="outside upper center", ncol=5, frameon=False, fontsize=9.5, labelcolor=TEXT)
fig.supxlabel("108 transfer-eval items (36 GPQA, 36 HLE, 36 MMLU-Pro). No control: 24 CoT samples per item, no instruction. Control: transfer-eval rows the organism complied on (strict pass, no mention),\n"
              "one sample per item and type; 'allow reasoning' = the 21 manytypes2 types minus ignore_question, decoy_topic, answer_hidden, misdirect. Base references are averaged over each bar's own items\n"
              "(24 samples each, with a CoT and with an empty analysis channel). n = rows. Error bars: 95% over item bootstraps.", fontsize=8.5, color=TEXT2)
fig.savefig(OUT, dpi=130, facecolor=SURFACE)
(RESULTS / "difficulty" / "control_accuracy_vs_base.json").write_text(json.dumps(summary, indent=1))
for k, v in summary.items():
    print(k, v)
print(OUT)
