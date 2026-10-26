"""CoT-control compliance against question difficulty, with and without controlling for CoT length, per organism.

Difficulty per item from item_difficulty.py (base gpt-oss-120b, 24 samples with a CoT and 24 without on the transfer eval's prompt): 1 - base CoT accuracy,
and 1 - base no-CoT accuracy. Compliance = strict pass with no instruction mention (nometa), per transfer-eval row (F0 wording, user turn), on the 21 manytypes2
types (10 trained by manytypes2, 11 held out). Per model, two logistic fits with type intercepts (L2 0.5 on them): pass ~ type + difficulty (raw) and
pass ~ type + difficulty + log2(output tokens) (length-controlled). Curves average the fitted probability over types; the length-controlled curve holds tokens at
the model's median on these rows. Bands: 2.5-97.5% over 300 bootstrap resamples of items. Also plots each item's median output tokens under the instructions.

    python difficulty_compliance.py --difficulty ../results/difficulty/gpqa_base.jsonl
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import controls

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS.parent.parent.parent / "figures/compliance_vs_difficulty_gpqa.png"
SURFACE, TEXT, TEXT2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
TYPES = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking", "alternating_case", "questions_only", "no_backtracking",
         "no_commitment", "decoy_topic", "end_of_sentence", "uppercase_thinking", "ignore_question", "no_question_restate", "no_option_text", "answer_hidden", "misdirect",
         "no_letter_e", "third_person", "end_apple", "no_calculation"]
MODELS = [("base", "gptoss-base", "#3987e5"), ("grpo3 s100", "gptoss-grpo3s100", "#d95926"), ("manytypes1b s150", "mt1b-s150", "#199e70"),
          ("manytypes2c s140", "manytypes2c-s140", "#d55181"), ("manytypes2e s170", "manytypes2e-s170", "#9085e9")]
GRID_X = np.linspace(0, 1, 41)


def rows_for(run, dataset):
    gens = [json.loads(l) for l in open(RESULTS / "transfer" / run / "gens.jsonl")]
    jd = {(r["key"], r["kind"]): r for r in map(json.loads, open(RESULTS / "transfer" / run / "judgments.jsonl"))}
    out = []
    for r in gens:
        if r["type"] not in TYPES or r["dataset"] != dataset or r["frame"] != "F0" or r["placement"] != "user":
            continue
        kind = "requirement" if r["type"] in controls.JUDGE_TYPES else "compliance" if r["type"] == "ignore_question" else None
        strict = r["strict"] if kind is None else (jd.get((r["key"], kind)) or {}).get("score")
        meta = jd.get((r["key"], "meta"))
        out.append(dict(item=r["item_idx"], type=r["type"], ok=float(bool(strict and meta is not None and meta.get("score") == 1)), tokens=max(r["n_output_tokens"], 1)))
    return out


def fit(X, y, n_types, l2=0.5):
    def nll(w):
        z = X @ w
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(w[:n_types] ** 2)
    def grad(w):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y)
        g[:n_types] += 2 * l2 * w[:n_types]
        return g
    return minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B").x


def curves(rows, diff, with_len, ref_tokens):
    types = sorted({r["type"] for r in rows})
    ti = {t: i for i, t in enumerate(types)}
    X = np.zeros((len(rows), len(types) + 1 + with_len))
    for k, r in enumerate(rows):
        X[k, ti[r["type"]]] = 1
        X[k, len(types)] = diff[r["item"]]
        if with_len:
            X[k, len(types) + 1] = np.log2(r["tokens"] / ref_tokens)
    w = fit(X, np.array([r["ok"] for r in rows]), len(types))
    a = w[:len(types)]
    return np.array([np.mean(1 / (1 + np.exp(-(a + w[len(types)] * x)))) for x in GRID_X]), w[len(types)]


def boot(rows, diff, with_len, ref_tokens, n=300, seed=0):
    rng = np.random.default_rng(seed)
    items = sorted({r["item"] for r in rows})
    by = {i: [r for r in rows if r["item"] == i] for i in items}
    cs, slopes = [], []
    for _ in range(n):
        pick = rng.choice(items, len(items))
        rs = [dict(r, item=(r["item"], j)) for j, i in enumerate(pick) for r in by[i]]
        d = {(i, j): diff[i] for j, i in enumerate(pick)}
        c, s = curves(rs, d, with_len, ref_tokens)
        cs.append(c)
        slopes.append(s)
    return np.percentile(cs, [2.5, 97.5], axis=0), np.percentile(slopes, [2.5, 97.5])


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10.5, color=TEXT, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT2)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT2)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.tick_params(colors=TEXT2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--difficulty", type=Path, required=True)
    ap.add_argument("--dataset", default="gpqa")
    args = ap.parse_args()
    diffs = [json.loads(l) for l in open(args.difficulty)]
    measures = [("1 - base accuracy with a CoT (24 samples)", {d["item_idx"]: 1 - d["cot_correct"] / d["samples"] for d in diffs}),
                ("1 - base accuracy without a CoT (24 samples)", {d["item_idx"]: 1 - d["nocot_correct"] / d["samples"] for d in diffs})]
    data = {label: rows_for(run, args.dataset) for label, run, _ in MODELS}
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), facecolor=SURFACE, layout="constrained")
    summary = {}
    for row, (mlabel, diff) in enumerate(measures):
        for label, run, color in MODELS:
            rows = data[label]
            ref = float(np.median([r["tokens"] for r in rows]))
            raw, s_raw = curves(rows, diff, False, ref)
            ctl, s_ctl = curves(rows, diff, True, ref)
            (lo_r, hi_r), ci_r = boot(rows, diff, False, ref)
            (lo_c, hi_c), ci_c = boot(rows, diff, True, ref)
            summary[f"{mlabel} | {label}"] = dict(slope_raw=round(float(s_raw), 3), slope_raw_ci=[round(float(x), 3) for x in ci_r], slope_len_controlled=round(float(s_ctl), 3),
                                                  slope_len_controlled_ci=[round(float(x), 3) for x in ci_c], median_tokens=ref, n_rows=len(rows))
            items = sorted({r["item"] for r in rows})
            per_item = [(diff[i], np.mean([r["ok"] for r in rows if r["item"] == i]), np.median([r["tokens"] for r in rows if r["item"] == i])) for i in items]
            ax = axes[row, 0]
            ax.scatter([p[0] for p in per_item], [100 * p[1] for p in per_item], s=14, color=color, alpha=0.45, zorder=3, linewidths=0)
            ax.plot(GRID_X, 100 * raw, color=color, lw=2, zorder=4, label=label)
            ax.fill_between(GRID_X, 100 * lo_r, 100 * hi_r, color=color, alpha=0.12, zorder=2, linewidth=0)
            ax = axes[row, 1]
            ax.plot(GRID_X, 100 * ctl, color=color, lw=2, zorder=4, label=label)
            ax.fill_between(GRID_X, 100 * lo_c, 100 * hi_c, color=color, alpha=0.12, zorder=2, linewidth=0)
            ax = axes[row, 2]
            ax.scatter([p[0] for p in per_item], [p[2] for p in per_item], s=16, color=color, alpha=0.7, zorder=3, linewidths=0, label=label)
            order = np.argsort([p[0] for p in per_item])
            xs, ys = np.array([p[0] for p in per_item])[order], np.log2([p[2] for p in per_item])[order]
            k = np.polyfit(xs, ys, 1)
            ax.plot(GRID_X, 2 ** np.polyval(k, GRID_X), color=color, lw=1.5, zorder=4)
        style(axes[row, 0], "Compliance, raw (dots: each item's pass rate over the 21 types)", mlabel + "  (harder to the right)", "strict pass, no mention (%)")
        style(axes[row, 1], "Compliance at each model's median CoT length (length-controlled)", mlabel + "  (harder to the right)", "strict pass, no mention (%)")
        style(axes[row, 2], "Output tokens under the instructions (item median over the 21 types)", mlabel + "  (harder to the right)", "tokens (log scale)")
        axes[row, 2].set_yscale("log", base=2)
        for c in (0, 1):
            axes[row, c].set_ylim(0, 100)
    fig.legend(*axes[0, 1].get_legend_handles_labels(), loc="outside upper center", ncol=len(MODELS), frameon=False, fontsize=10.5, labelcolor=TEXT)
    fig.supxlabel(f"{args.dataset.upper()} items of the transfer eval (n=36), F0 wording, instruction in the user turn, 21 types (manytypes2's 10 trained and 11 held out), one sample per item and type. "
                  "Fits: logistic, type intercepts, difficulty linear, plus log2 tokens for the controlled fit; bands 95% over item bootstraps.", fontsize=8.5, color=TEXT2)
    fig.savefig(OUT, dpi=130, facecolor=SURFACE)
    out_json = RESULTS / "difficulty" / f"compliance_vs_difficulty_{args.dataset}.json"
    out_json.write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(k, v)
    print(OUT)


if __name__ == "__main__":
    main()
