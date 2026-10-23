"""Tables for prompt_wordsupp.py. Pass one or more rows.jsonl files; --json DIR also writes the tables as JSON.

    uv run python prompt_wordsupp_analyze.py ../results/prompt_wordsupp/rows.jsonl ../results/prompt_wordsupp_nm/rows.jsonl
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("rows", nargs="+", type=Path)
ap.add_argument("--json", type=Path, help="write summary/by_dataset/paired JSON here")
args = ap.parse_args()

CONDS = ["none", "reference", "explainer", "explainer_nm"]
df = pd.concat([pd.read_json(p, lines=True) for p in args.rows], ignore_index=True)
df["no_cot"] = df.reasoning.fillna("").str.strip() == ""
df["clean_sub"] = (df.cot_kw_sub == 0) & ~df.no_cot      # explainer's rule: no case-insensitive substring match
df["clean_word"] = (df.cot_kw_word == 0) & ~df.no_cot    # reference matcher, keyword only
df["ans_clean"] = df.ans_kw_sub == 0
df["trunc"] = df.finish == "length"
# a violation that restates the rule: a negation or "forbidden" within 60 characters before the keyword
REMIND = r"(must not|mustn't|can't|cannot|avoid|forbidden|not (to )?(use|write|say|mention))[^.\n]{0,60}"
df["restates"] = [r.cot_kw_sub > 0 and bool(re.search(REMIND + re.escape(r.keyword), r.reasoning or "", re.I)) for r in df.itertuples()]
rng = np.random.default_rng(0)


def ci(x: np.ndarray, n=4000) -> tuple[float, float]:
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


rows = []
for (m, c), g in df.groupby(["model", "cond"], sort=False):
    lo, hi = ci(g.clean_sub.to_numpy(float))
    rows.append(dict(model=m, cond=c, n=len(g), clean_sub=round(g.clean_sub.mean(), 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                     clean_word=round(g.clean_word.mean(), 3), ref_strict=round(g.ref_strict.mean(), 3),
                     median_kw=float(g.cot_kw_sub.median()), ans_clean=round(g.ans_clean.mean(), 3), acc=round(g.correct.mean(), 3),
                     violating=int((g.cot_kw_sub > 0).sum()), restating=int(g.restates.sum()),
                     trunc=int(g.trunc.sum()), empty=int(g.no_cot.sum()), med_tok=float(g.completion_tokens.median()),
                     cost=round(g.cost.sum(), 2)))
t = pd.DataFrame(rows)
t["cond"] = pd.Categorical(t.cond, CONDS)
t = t.sort_values(["model", "cond"])
print(t.to_string(index=False))

print("\nper dataset, clean_sub")
bd = df.pivot_table(index=["model", "dataset"], columns="cond", values="clean_sub", aggfunc="mean").round(3)
print(bd.to_string())

print("\npaired per item (same item, same model), difference in clean_sub")
paired = []
for m, g in df.groupby("model"):
    p = g.pivot_table(index=["dataset", "item_idx"], columns="cond", values="clean_sub", aggfunc="mean").dropna()
    for a, b in [("explainer", "reference"), ("explainer", "none"), ("reference", "none"), ("explainer_nm", "explainer"), ("explainer_nm", "reference")]:
        if a not in p or b not in p:
            continue
        d = (p[a] - p[b]).to_numpy(float)
        lo, hi = ci(d)
        paired.append(dict(model=m, comparison=f"{a} - {b}", diff=round(d.mean(), 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3), n=len(d)))
        print(f"  {m:13s} {a} - {b}: {d.mean():+.3f} [{lo:.3f}, {hi:.3f}]  n={len(d)}")
print(f"\ntotal cost ${df.cost.sum():.2f}; providers: {df.groupby('model').provider.value_counts().to_dict()}")

# Partial compliance. partial = 1/(1+hits), grpo.py's word_suppression partial; per_1k = keyword hits per 1000 CoT
# words ([A-Za-z]+ runs); rel = per_1k / the same model's no-constraint per_1k on the same item (items whose
# unconstrained CoT uses the keyword).
df["partial"] = 1 / (1 + df.cot_kw_sub)
df["words"] = df.reasoning.fillna("").map(lambda s: len(re.findall(r"[A-Za-z]+", s)))
df["per_1k"] = 1000 * df.cot_kw_sub / df.words.clip(lower=1)
base = df[df.cond == "none"].set_index(["model", "dataset", "item_idx"]).per_1k
df["rel"] = [r.per_1k / base[(r.model, r.dataset, r.item_idx)] if base.get((r.model, r.dataset, r.item_idx), 0) > 0 else np.nan
             for r in df.itertuples()]
print("\npartial compliance (means; 95% CIs bootstrap items)")
partial = []
for (m, c), g in df.groupby(["model", "cond"], sort=False):
    plo, phi = ci(g.partial.to_numpy(float))
    klo, khi = ci(g.per_1k.to_numpy(float))
    rel = g.rel.dropna()
    partial.append(dict(model=m, cond=c, partial=round(g.partial.mean(), 3), partial_lo=round(plo, 3), partial_hi=round(phi, 3),
                        per_1k=round(g.per_1k.mean(), 2), per_1k_lo=round(klo, 2), per_1k_hi=round(khi, 2),
                        median_per_1k=round(g.per_1k.median(), 2), median_words=float(g.words.median()),
                        rel_median=round(rel.median(), 3), below_base=round((rel < 1).mean(), 3), n_rel=len(rel)))
pt = pd.DataFrame(partial)
pt["cond"] = pd.Categorical(pt.cond, CONDS)
pt = pt.sort_values(["model", "cond"])
print(pt.to_string(index=False))
print("\npaired per item, partial and per_1k")
pdiff = []
for m, g in df.groupby("model"):
    for metric in ("partial", "per_1k"):
        p = g.pivot_table(index=["dataset", "item_idx"], columns="cond", values=metric, aggfunc="mean").dropna()
        for a, b in [("reference", "none"), ("explainer", "none"), ("explainer_nm", "none"), ("explainer", "reference"), ("explainer_nm", "reference")]:
            if a not in p or b not in p:
                continue
            d = (p[a] - p[b]).to_numpy(float)
            lo, hi = ci(d)
            pdiff.append(dict(model=m, metric=metric, comparison=f"{a} - {b}", diff=round(d.mean(), 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3), n=len(d)))
            print(f"  {m:13s} {metric:7s} {a} - {b}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

if args.json:
    args.json.mkdir(parents=True, exist_ok=True)
    (args.json / "summary.json").write_text(json.dumps(t.assign(cond=t.cond.astype(str)).to_dict("records"), indent=1))
    long = bd.reset_index().melt(id_vars=["model", "dataset"], var_name="cond", value_name="clean_sub").dropna()
    (args.json / "by_dataset.json").write_text(json.dumps(long.to_dict("records"), indent=1))
    (args.json / "paired.json").write_text(json.dumps(paired, indent=1))
    (args.json / "partial.json").write_text(json.dumps(pt.assign(cond=pt.cond.astype(str)).to_dict("records"), indent=1))
    (args.json / "partial_paired.json").write_text(json.dumps(pdiff, indent=1))
