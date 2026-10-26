"""Compare reasoning length with vs without the CoT-Control requirement, per model, and its relation to NCRI.
Instructed = Neel's graded Block-A traces for the same 36 questions (3 instructions each); baseline = rows.jsonl here.
Run: ~/code/results/neel-astra/.venv/bin/python analyze.py"""
import json, os
import pandas as pd, numpy as np
from scipy.stats import spearmanr
os.chdir(os.path.dirname(os.path.abspath(__file__)))
R = "/home/john/code/results/neel-astra"
base = pd.DataFrame([json.loads(l) for l in open("rows.jsonl")]); base = base[base.status == "ok"]
base["cap"] = base.finish == "length"; base["q"] = base.dataset + ":" + base.qid.astype(str)
it = pd.read_csv(f"{R}/work/sol-ncri/confounds/items.csv"); it = it[it.model.isin(base.model.unique())]
it["q"] = it.item_id.str.split(":").str[1] + ":" + it.item_id.str.split(":").str[2]
nc = pd.read_csv(f"{R}/work/sol-ncri/verify/models.csv").set_index("model").new_ncri
per = pd.DataFrame({"ncri": nc.reindex(base.model.unique()),
                    "base_median": base.groupby("model").chars.median(), "instr_median": it.groupby("model").chars.median(),
                    "base_cap": base.groupby("model").cap.mean(), "instr_cap": it.groupby("model").cap.mean(),
                    "base_acc": base.groupby("model").correct.mean(), "instr_acc": it.groupby("model").correct.mean(),
                    "n_base": base.groupby("model").size(), "providers_off_pin": base.assign(off=base.provider != base.provider_pin).groupby("model").off.sum()})
# per question ratio instructed/baseline (instructed = mean over its 3 instruction types)
iq = it.groupby(["model", "q"]).chars.mean(); bq = base.set_index(["model", "q"]).chars
ratio = (np.log2((iq + 1) / (bq.reindex(iq.index) + 1))).dropna()
per["log2_instr_over_base"] = ratio.groupby(level=0).median()
per = per.sort_values("ncri", ascending=False); print(per.round(3).to_string())
def rho(col): r, p = spearmanr(per.ncri, per[col]); return f"{r:+.2f} (p={p:.3f})"
for c in ["base_median", "instr_median", "base_cap", "instr_cap", "base_acc", "log2_instr_over_base"]: print(f"NCRI vs {c:22s} {rho(c)}")
nocap = base[~base.cap].groupby("model").chars.median().reindex(per.index); r, p = spearmanr(per.ncri, nocap); print(f"NCRI vs base_median (no cap-outs)  {r:+.2f} (p={p:.3f})")
r, p = spearmanr(per.base_median, per.instr_median); print(f"baseline vs instructed median length across models {r:+.2f} (p={p:.3f})")
qs = []
for q, g in base.groupby("q"):
    g = g.set_index("model"); qs.append(spearmanr(nc[g.index], g.chars)[0])
qs = np.array(qs); print("per-question rho NCRI vs baseline chars: median", np.nanmedian(qs).round(2), "share negative", np.mean(qs < 0).round(2))
per.to_csv("per_model.csv")
