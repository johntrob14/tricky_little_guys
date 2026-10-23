"""Per-condition summary of a pulled run dir, after grade.sh's grade / judge / analyze steps.

    python summarize.py RUN_DIR     # writes RUN_DIR/summary.csv and summary_constraints.csv, prints both

strict_adj (the default metric) = the length-adjusted strict pass count: the reference grader's strict pass rate with
meta-reasoning allowed, minus the change predicted by the condition's CoT lengths alone under a logistic fit on the unsteered
rows (lenadj.py), times n. strict = the raw count (analysis/records.csv strict); strict_nometa = raw with meta-discussion
excluded (the judge flags CoTs that talk about the rule). partial = the
grpo reward's partial compliance on the extracted CoT (rl/grpo.py compliance; none for ignore_question). Length: output
tokens and CoT words per row (medians); truncated = finish_reason length. banned per 1k = banned-word hits per 1,000 CoT
words over the two suppression constraints.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grading"))
sys.path.insert(0, str(HERE.parent / "rl"))
from grpo import compliance  # noqa: E402
from lenadj import add_length_adjustment, adjusted_rate  # noqa: E402

SUPPRESSION = ["word_suppression", "multiple_word_suppression"]


def partial(row):
    c, ga, text = row["constraint"], row["grader_args"], row["reasoning"] or ""
    if c == "ignore_question":
        return None
    if not text.strip():
        return 0.0
    kw = {"words": ga.get("keywords"), "sentence": ga.get("target_sentence"), "word": ga.get("target_word")}
    return compliance(c, text, kw)[0]


def banned_hits(row):
    text = (row["reasoning"] or "").lower()
    return sum(len(re.findall(rf"\b{re.escape(w.lower())}\b", text)) for w in row["grader_args"].get("keywords") or [])


run = Path(sys.argv[1])
rec = pd.read_csv(run / "analysis/records.csv")
gens = {}
for f in (run / "generations").glob("*/astra.jsonl"):
    for line in open(f):
        r = json.loads(line)
        gens[(r["condition"], r["key"])] = r
rows = [gens[(c, k)] for c, k in zip(rec.condition, rec.key)]
rec["partial"] = [partial(r) for r in rows]
rec["cot_words"] = [len((r["reasoning"] or "").split()) for r in rows]
rec["banned"] = [banned_hits(r) if r["constraint"] in SUPPRESSION else None for r in rows]
rec["strict_nometa"] = rec.strict_nometa.fillna(False).astype(bool)
rec["strict"] = rec.strict.fillna(False).astype(bool)
rec["correct"] = rec.correct.fillna(False).astype(bool)
rec["truncated"] = rec.finish_reason == "length"
rec["seed"] = rec.condition.str.extract(r"_s(\d+)$")[0].fillna("0").astype(int)
rec["condition"] = rec.condition.str.replace(r"_s\d+$", "", regex=True)  # seeds pooled per condition
base = next(c for c in rec.condition.unique() if c.endswith("unsteered"))
odds_per_doubling = add_length_adjustment(rec, base)
uns = rec[rec.condition == base]


def agg(g):
    sup = g[g.constraint.isin(SUPPRESSION)]
    adj_seed = "/".join(str(int(round(adjusted_rate(gs, uns[uns.seed == s]) * len(gs)))) for s, gs in g.groupby("seed"))
    return pd.Series({"n": len(g), "seeds": g.seed.nunique(), "strict_adj": int(round(adjusted_rate(g, uns) * len(g))), "strict_adj_per_seed": adj_seed,
                      "strict": int(g.strict.sum()), "strict_per_seed": "/".join(str(int(x)) for x in g.groupby("seed").strict.sum()),
                      "strict_nometa": int(g.strict_nometa.sum()), "strict_nometa_per_seed": "/".join(str(int(x)) for x in g.groupby("seed").strict_nometa.sum()),
                      "partial": g.partial.mean(), "correct": int(g.correct.sum()), "truncated": int(g.truncated.sum()),
                      "empty_cot": int((g.cot_words == 0).sum()), "median_output_tokens": g.n_output_tokens.median(),
                      "median_cot_words": g.cot_words.median(), "banned_per_1k_cot_words": 1000 * sup.banned.sum() / max(sup.cot_words.sum(), 1)})


summary = rec.groupby("condition").apply(agg)
per_c = rec.groupby(["condition", "constraint"]).agg(strict=("strict", "sum"), strict_nometa=("strict_nometa", "sum"), partial=("partial", "mean"),
                                                      median_cot_words=("cot_words", "median"), n=("strict", "size")).reset_index()
per_c["strict_adj"] = [round(adjusted_rate(rec[(rec.condition == c) & (rec.constraint == k)], uns[uns.constraint == k]) * n, 1)
                       for c, k, n in zip(per_c.condition, per_c.constraint, per_c.n)]
summary_note = f"length fit on {len(uns)} unsteered rows: strict odds x{odds_per_doubling:.2f} per doubling of CoT words"
summary.to_csv(run / "summary.csv")
per_c.to_csv(run / "summary_constraints.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
print(summary_note)
print(summary.round(3))
print(per_c.pivot(index="constraint", columns="condition", values="strict"))
print(per_c.pivot(index="constraint", columns="condition", values="median_cot_words"))
