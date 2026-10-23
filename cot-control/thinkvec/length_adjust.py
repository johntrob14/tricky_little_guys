"""Length confound: how strict compliance depends on CoT length in the organism's own unsteered completions, and each steered
condition's strict rate with the part predicted by its length shift removed.

    python length_adjust.py RUN_DIR OUT_PNG

Model: logistic regression of the reference grader's strict pass (meta-reasoning allowed) on log2(CoT words + 1) with a
constraint intercept, fit on the unsteered rows only (all seeds). For a condition c, predicted_c = mean over its rows of
P(pass | constraint, its CoT length); length-adjusted_c = observed_c - (predicted_c - predicted_unsteered). Writes
RUN_DIR/length_adjust.json. Also prints the base model's partial-compliance vs length correlation (results/thinkvec/main).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

matplotlib.use("Agg")
R = Path(__file__).resolve().parent.parent / "results/thinkvec"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grading"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl"))
from grpo import compliance  # noqa: E402


def partial(row):
    ga, text = row["grader_args"], row["reasoning"] or ""
    if row["constraint"] == "ignore_question":
        return None
    if not text.strip():
        return 0.0
    return compliance(row["constraint"], text, {"words": ga.get("keywords"), "sentence": ga.get("target_sentence"), "word": ga.get("target_word")})[0]
run, out = Path(sys.argv[1]), Path(sys.argv[2])


def load(run_dir):
    rec = pd.read_csv(run_dir / "analysis/records.csv")
    rec["seed"] = rec.condition.str.extract(r"_s(\d+)$")[0].fillna("0").astype(int)
    gens = {}
    for f in (run_dir / "generations").glob("*/astra.jsonl"):
        for line in open(f):
            r = json.loads(line)
            gens[(r["condition"], r["key"])] = r
    rec["cot_words"] = [len((gens[(c, k)]["reasoning"] or "").split()) for c, k in zip(rec.condition, rec.key)]
    rec["condition"] = rec.condition.str.replace(r"_s\d+$", "", regex=True)
    rec["strict"] = rec.strict.fillna(False).astype(bool)
    rec["correct"] = rec.correct.fillna(False).astype(bool)
    rec["loglen"] = np.log2(rec.cot_words + 1)
    return rec


rec = load(run)
base = next(c for c in rec.condition.unique() if c.endswith("unsteered"))
cons = sorted(rec.constraint.unique())
uns = rec[rec.condition == base]


def design(df):
    X = np.zeros((len(df), len(cons) + 1))
    for j, c in enumerate(cons):
        X[(df.constraint == c).values, j] = 1
    X[:, -1] = df.loglen.values
    return X


def fit(df, y):
    X = design(df)
    def nll(w):
        z = X @ w
        return np.sum(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)) + 0.01 * np.sum(w[:-1] ** 2)
    return minimize(nll, np.zeros(X.shape[1]), method="L-BFGS-B").x


w = fit(uns, uns.strict.values.astype(float))
slope = w[-1]
p = lambda df: 1 / (1 + np.exp(-(design(df) @ w)))
# per-constraint slopes (separate fits on the unsteered rows, where both outcomes occur)
per_c = {}
for c in cons:
    d = uns[uns.constraint == c]
    if 0 < d.strict.sum() < len(d):
        wc = minimize(lambda v: np.sum(np.logaddexp(0, -(v[0] + v[1] * d.loglen.values)) * d.strict.values + np.logaddexp(0, v[0] + v[1] * d.loglen.values) * (1 - d.strict.values)) + 0.01 * v[1] ** 2,
                      np.zeros(2), method="L-BFGS-B").x
        per_c[c] = {"odds_ratio_per_doubling": float(np.exp(wc[1])), "n_pass": int(d.strict.sum()), "n": len(d), "median_cot_words": float(d.cot_words.median())}
pred_uns = p(uns).mean()
rows = []
for c, g in rec.groupby("condition"):
    obs = g.strict.mean()
    pred = p(g).mean()
    rows.append({"condition": c, "n": len(g), "observed": obs, "predicted_from_length": pred, "length_adjusted": obs - (pred - pred_uns),
                 "median_cot_words": float(g.cot_words.median()), "strict": int(g.strict.sum())})
tab = pd.DataFrame(rows).set_index("condition")
res = {"shared_slope_odds_ratio_per_doubling_of_cot_words": float(np.exp(slope)), "unsteered_rows": len(uns),
       "per_constraint": per_c, "conditions": tab.to_dict(orient="index"),
       "note": "logistic fit on the organism's unsteered rows: strict ~ constraint + log2(cot_words + 1); adjusted = observed - (length-predicted - unsteered-predicted)"}

# base model: partial compliance vs length (its strict is 0 everywhere)
bm = pd.read_csv(R / "main/analysis/records.csv")
bgens = {}
for f in (R / "main/generations").glob("*/astra.jsonl"):
    for line in open(f):
        r = json.loads(line)
        bgens[(r["condition"], r["key"])] = r
bm["cot_words"] = [len((bgens[(c, k)]["reasoning"] or "").split()) for c, k in zip(bm.condition, bm.key)]
bm["partial"] = [partial(bgens[(c, k)]) for c, k in zip(bm.condition, bm.key)]
bu = bm[bm.partial.notna()]  # all three base conditions (steering only changes length there), 288 rows
sp = {c: float(pd.Series(d.partial.values).corr(pd.Series(np.log2(d.cot_words.values + 1)), method="spearman")) for c, d in bu.groupby("constraint")}
res["base_model"] = {"strict": int(bm.strict.fillna(False).astype(bool).sum()), "n": len(bm), "n_with_partial": len(bu),
                     "spearman_partial_vs_log_cot_words_pooled": float(pd.Series(bu.partial.values).corr(pd.Series(np.log2(bu.cot_words.values + 1)), method="spearman")),
                     "spearman_partial_vs_log_cot_words_per_constraint": sp}
(run / "length_adjust.json").write_text(json.dumps(res, indent=1))
pd.set_option("display.width", 200)
print(f"shared slope: odds ratio per doubling of CoT words = {np.exp(slope):.2f} (fit on {len(uns)} unsteered organism rows)")
print({c: round(v["odds_ratio_per_doubling"], 2) for c, v in per_c.items()})
pd.set_option("display.max_columns", None)
print(tab.round(3).to_string())
print("base model (all 3 conditions, 324 rows):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["base_model"].items() if k != "spearman_partial_vs_log_cot_words_per_constraint"})
print("  per constraint:", {k: round(v, 2) for k, v in sp.items()})

SURFACE, TEXT, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#33332f"
C_OBS, C_ADJ, C_PRED, C_REF = "#d95926", "#3987e5", "#c3c2b7", "#199e70"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": TEXT,
                     "ytick.color": MUTED, "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1, 1.6]})
ax = axes[0]
bins = np.quantile(uns.loglen, np.linspace(0, 1, 9))
idx = np.clip(np.digitize(uns.loglen, bins[1:-1]), 0, 7)
for b in range(8):
    d = uns[idx == b]
    ax.scatter([2 ** d.loglen.mean()], [d.strict.mean()], s=30 + 3 * len(d), color=C_OBS, zorder=3, edgecolors=SURFACE)
    ax.annotate(f"{int(d.strict.sum())}/{len(d)}", (2 ** d.loglen.mean(), d.strict.mean()), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7.5)
xs = np.linspace(uns.loglen.min(), uns.loglen.max(), 100)
mix = uns.constraint.value_counts(normalize=True)
curve = sum(mix[c] / (1 + np.exp(-(w[cons.index(c)] + slope * xs))) for c in cons)
ax.plot(2 ** xs, curve, color=C_ADJ, linewidth=2)
ax.set_xscale("log", base=2)
ax.set_xlabel("CoT words (log scale)"); ax.set_ylabel("strict pass rate (meta-reasoning allowed)")
ax.set_ylim(0, 1)
ax.set_title(f"Organism unsteered, {len(uns)} rows: pass rate by CoT-length octile\nfit: odds x{np.exp(slope):.2f} per doubling of length (constraint intercepts)", loc="left", fontsize=10)
ax = axes[1]
def parse(c):
    m = re.fullmatch(r"(?:org_)?([a-z_]+?)_(L[\d-]+(?:s\d+)?)_(p|m)([\d.]+)", c)
    return None if m is None else (m[1], m[2], (1 if m[3] == "p" else -1) * float(m[4]))
parsed = {c: parse(c) for c in tab.index if c != base}
main_tag = max({p[1] for p in parsed.values()}, key=lambda t: sum(p[1] == t for p in parsed.values()))
alphas = sorted({p[2] for p in parsed.values() if p[1] == main_tag} | {0.0})
others = sorted({(p[1], p[2]) for p in parsed.values() if p[1] != main_tag})
cats = [o for o in others if o[1] < 0] + alphas + [o for o in others if o[1] > 0]
xpos = {cat: i for i, cat in enumerate(cats)}
x_of = lambda c: xpos[0.0] if c == base else (xpos[parsed[c][2]] if parsed[c][1] == main_tag else xpos[(parsed[c][1], parsed[c][2])])
for c in tab.index:
    is_rand = "rand" in c
    dx = 0.18 if is_rand else 0
    mk = "s" if is_rand else ("D" if c != base and parsed[c][1] != main_tag else "o")
    x = x_of(c) + dx
    ax.plot([x, x], [tab.loc[c, "observed"], tab.loc[c, "length_adjusted"]], color=MUTED, linewidth=1, zorder=1)
    ax.scatter([x], [tab.loc[c, "predicted_from_length"]], marker=mk, s=70, facecolors="none", edgecolors=C_PRED, linewidths=1.5, zorder=2)
    ax.scatter([x], [tab.loc[c, "observed"]], marker=mk, s=80, color=C_OBS, zorder=3, edgecolors=SURFACE)
    ax.scatter([x], [tab.loc[c, "length_adjusted"]], marker=mk, s=80, color=C_ADJ, zorder=4, edgecolors=SURFACE)
    ax.annotate(f"{tab.loc[c, 'length_adjusted']:.2f}", (x, tab.loc[c, "length_adjusted"]), textcoords="offset points", xytext=(10, 0), ha="left", va="center", fontsize=7.5, color=C_ADJ)
ax.axhline(tab.loc[base, "observed"], color=C_REF, linestyle=":", linewidth=1.2)
ticks = [f"{t.replace('L', 'block ')}\nalpha {a:+g}" if isinstance(cat, tuple) else ("0\n(unsteered)" if a == 0 else f"{a:+g}") for cat in cats for (t, a) in [cat if isinstance(cat, tuple) else (None, cat)]]
ax.set_xticks(range(len(cats)), ticks, fontsize=8.5); ax.set_xlim(-0.6, len(cats) - 0.4); ax.set_ylim(0, 1)
ax.set_title("Strict compliance per condition: observed, predicted from its CoT lengths alone, and length-adjusted", loc="left", fontsize=10)
handles = [plt.Line2D([], [], color=C_OBS, marker="o", markersize=8, linewidth=0, label="observed"),
           plt.Line2D([], [], color=C_PRED, marker="o", markersize=8, linewidth=0, markerfacecolor="none", label="predicted from length alone (unsteered fit)"),
           plt.Line2D([], [], color=C_ADJ, marker="o", markersize=8, linewidth=0, label="length-adjusted = observed - (predicted - unsteered predicted)"),
           plt.Line2D([], [], color=C_REF, linestyle=":", label="unsteered observed")]
ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5, labelcolor=TEXT)
vec_name = next(p[0] for p in parsed.values() if not p[0].startswith("rand"))
fig.suptitle(f"Length confound in the organism steering results ({vec_name} vector, run {run.name}); circles = main layer set, squares = random direction, diamonds = other layer set", color=TEXT, x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
