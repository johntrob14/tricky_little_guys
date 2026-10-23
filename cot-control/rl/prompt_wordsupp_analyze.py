"""Tables for prompt_wordsupp.py.

    uv run python prompt_wordsupp_analyze.py ../results/prompt_wordsupp/rows.jsonl
"""

import sys

import numpy as np
import pandas as pd

df = pd.read_json(sys.argv[1], lines=True)
df["no_cot"] = df.reasoning.fillna("").str.strip() == ""
df["clean_sub"] = (df.cot_kw_sub == 0) & ~df.no_cot      # explainer's rule: no case-insensitive substring match
df["clean_word"] = (df.cot_kw_word == 0) & ~df.no_cot    # reference matcher, keyword only
df["ans_clean"] = df.ans_kw_sub == 0
df["trunc"] = df.finish == "length"
rng = np.random.default_rng(0)


def ci(x: np.ndarray, n=4000) -> str:
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"


rows = []
for (m, c), g in df.groupby(["model", "cond"], sort=False):
    rows.append(dict(model=m, cond=c, n=len(g), clean_sub=round(g.clean_sub.mean(), 3), ci=ci(g.clean_sub.to_numpy(float)),
                     clean_word=round(g.clean_word.mean(), 3), ref_strict=round(g.ref_strict.mean(), 3),
                     median_kw=g.cot_kw_sub.median(), ans_clean=round(g.ans_clean.mean(), 3), acc=round(g.correct.mean(), 3),
                     trunc=int(g.trunc.sum()), empty=int(g.no_cot.sum()), med_tok=g.completion_tokens.median(), cost=round(g.cost.sum(), 2)))
t = pd.DataFrame(rows)
t["cond"] = pd.Categorical(t.cond, ["none", "reference", "explainer", "explainer_nm"])
print(t.sort_values(["model", "cond"]).to_string(index=False))

print("\nper dataset, clean_sub")
print(df.pivot_table(index=["model", "dataset"], columns="cond", values="clean_sub", aggfunc="mean").round(3).to_string())

print("\npaired per item (same item, same model), difference in clean_sub")
for m, g in df.groupby("model"):
    p = g.pivot_table(index=["dataset", "item_idx"], columns="cond", values="clean_sub", aggfunc="mean").dropna()
    for a, b in [("explainer", "reference"), ("explainer", "none"), ("reference", "none"), ("explainer_nm", "explainer"), ("explainer_nm", "reference")]:
        d = (p[a] - p[b]).to_numpy(float)
        print(f"  {m:13s} {a} - {b}: {d.mean():+.3f} {ci(d)}  n={len(d)}")
    used = g[(g.cond == "none") & (g.cot_kw_sub > 0)].set_index(["dataset", "item_idx"]).index
    sub = p.loc[p.index.isin(used)]
    print(f"  {m:13s} items whose unconstrained CoT uses the keyword: {len(sub)}/{len(p)}; clean_sub there: "
          + ", ".join(f"{c} {sub[c].mean():.3f}" for c in ["reference", "explainer", "explainer_nm"]))
print(f"\ntotal cost ${df.cost.sum():.2f}; providers: {df.groupby('model').provider.value_counts().to_dict()}")
