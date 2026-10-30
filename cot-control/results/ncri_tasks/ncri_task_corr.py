"""Which NCRI 15.2 task families (effective domains) and rungs correlate, across models, with CoT controllability?
Per model: no-CoT accuracy per domain and per rung over the 64 sealed rungs only (every model saw the same items), from the item-level data
in the private HF dataset latentMO/ncri-15.2 (raw/ncri15_2_item_level.csv.gz, raw/rungs_ncri15_2.csv, raw/models_ncri15_2.csv).
CoT-control measures (base models only):
  neel_matched  Neel's 34 open-weight models, 108 Astra items, strict pass reweighted to the pooled reasoning-length distribution (outctl/neel_len/neel_length.json)
  neel_raw      the same 34 models, raw strict pass, ignore_question excluded (neel-astra/dash/neel_models.csv)
  john_written  John's runs on the 72 held-out Astra items, prompt as written (outctl/elicit/dash/compare_long.json on john/inksm-sft)
  john_elicited the same with John's elicitation prompt
For each (measure, feature): Spearman rho with a two-sided permutation p (10k), Benjamini-Hochberg q within (measure, level), partial Spearman
controlling for overall NCRI (rank residuals, permutation p), and for the Neel measures rho without the 7 GLM models.
.venv: cot-control/rl/.venv/bin/python ncri_task_corr.py <ncri dir with raw/> <compare_long.json>  -> ncri_task_corr.json"""
import csv, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata

D, CL = sys.argv[1], sys.argv[2]
H = os.path.dirname(os.path.abspath(__file__))
R = np.random.default_rng(0)
NPERM = 10000

rungs = pd.read_csv(f"{D}/raw/rungs_ncri15_2.csv")
sealed = set(rungs[rungs.kind == "sealed"].rung_id)
items = pd.read_csv(f"{D}/raw/ncri15_2_item_level.csv.gz")
items = items[items.rung.isin(sealed)]
dom_acc = items.groupby(["model", "effective_domain"]).correct.mean().unstack()
rung_acc = items.groupby(["model", "rung"]).correct.mean().unstack()
ncri = pd.read_csv(f"{D}/raw/models_ncri15_2.csv").set_index("slug").ncri15_2

nl = json.load(open(f"{H}/../outctl/neel_len/neel_length.json"))["models"]
neel_matched = {m["model"].replace("/", "_"): m["matched"] for m in nl}
neel_raw = {r["model"].replace("/", "_"): 100 * float(r["cot_control"]) for r in csv.DictReader(open(f"{H}/../neel-astra/dash/neel_models.csv"))}
SLUG = {"gpt-oss-120b": "openai_gpt-oss-120b", "Qwen3.5-397B": "qwen_qwen3.5-397b-a17b", "GLM-5.3": "z-ai_glm-5.3", "Qwen3.8-27B": "qwen_qwen3.8-27b",
        "Inkling-Small": "thinkingmachines_inkling-small", "Qwen3.5-4B": "local_qwen3.5-4b", "Qwen3.5-9B": "qwen_qwen3.5-9b",
        "Nemotron-3 Ultra": "nvidia_nemotron-3-ultra-550b-a55b", "Nemotron-3 Nano": "nvidia_nemotron-3-nano-30b-a3b", "DeepSeek V4 Flash": "deepseek_deepseek-v4-flash",
        "Qwen3-32B": "qwen_qwen3-32b", "GLM-5.2": "z-ai_glm-5.2", "Claude Sonnet 4.5": "anthropic_claude-sonnet-4.5", "DeepSeek V3.2 exp": "deepseek_deepseek-v3.2-exp",
        "Kimi-K3": "moonshotai_kimi-k3", "GPT-5.1": "openai_gpt-5.1", "GPT-5.2": "openai_gpt-5.2", "GPT-5": "openai_gpt-5", "GPT-6 Astra": "openai_gpt-6-astra",
        "gpt-oss-20b": "openai_gpt-oss-20b", "Gemma 4 31B": "google_gemma-4-31b-it"}
john = {"CoT control, as written": {}, "CoT control, elicited": {}}
for r in json.load(open(CL)):
    if r["group"] == "Base model" and r["measure"] in john and r["model"] in SLUG: john[r["measure"]][SLUG[r["model"]]] = r["value"]
MEAS = {"neel_matched": neel_matched, "neel_raw": neel_raw, "john_written": john["CoT control, as written"], "john_elicited": john["CoT control, elicited"]}
for k, v in MEAS.items():
    missing = [m for m in v if m not in dom_acc.index or m not in ncri.index]
    assert not missing, (k, missing)

def resid(a, z):
    b = np.polyfit(z, a, 1)
    return a - np.polyval(b, z)

def corr_perm(x, y):
    zs = lambda a: (a - a.mean()) / a.std()
    xs, ys = zs(x), zs(y)
    r = float(xs @ ys / len(x))
    perm = R.permuted(np.tile(ys, (NPERM, 1)), axis=1) @ xs / len(x)
    return r, float((1 + np.sum(np.abs(perm) >= abs(r) - 1e-12)) / (NPERM + 1))

def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out

out = {}
for mk, mv in MEAS.items():
    models = sorted(mv)
    rho_ncri, p_ncri = corr_perm(rankdata(ncri[models].values), rankdata([mv[m] for m in models]))
    out[mk] = {"n_models": len(models), "models": models, "ncri": {"rho": rho_ncri, "p": p_ncri}}
    for level, A in [("domain", dom_acc), ("rung", rung_acc)]:
        rows = []
        for f in A.columns:
            ms = [m for m in models if not np.isnan(A.loc[m, f])]  # GLM-5.3 has no cfg or o_gsm1k rows
            xa, ya = A.loc[ms, f].values, np.array([mv[m] for m in ms])
            if np.std(xa) == 0: continue
            x, y, z = rankdata(xa), rankdata(ya), rankdata(ncri[ms].values)
            g = np.array([m.startswith("z-ai_") for m in ms])
            rho, p = corr_perm(x, y)
            prho, pp = corr_perm(resid(x, z), resid(y, z))
            row = dict(feature=f, n=len(ms), rho=rho, p=p, partial_rho=prho, partial_p=pp, mean_acc=float(xa.mean()))
            if mk.startswith("neel"): row["rho_no_glm"] = float(np.corrcoef(rankdata(xa[~g]), rankdata(ya[~g]))[0, 1])
            rows.append(row)
        for key in ("p", "partial_p"):
            for r_, q in zip(rows, bh([r_[key] for r_ in rows])): r_[key.replace("p", "q") if key == "p" else "partial_q"] = float(q)
        rows.sort(key=lambda r_: -r_["rho"])
        out[mk][level] = rows

json.dump(out, open(f"{H}/ncri_task_corr.json", "w"), indent=1)
for mk in MEAS:
    o = out[mk]
    print(f"\n### {mk}  n={o['n_models']}  overall NCRI rho {o['ncri']['rho']:+.2f} (p {o['ncri']['p']:.3f})")
    for level in ("domain", "rung"):
        rows = o[level] if level == "domain" else o[level][:8] + o[level][-3:]
        print(f"  {level}: feature, rho, p, q, partial rho (NCRI), partial p, partial q, rho without GLM, mean accuracy")
        for r_ in rows:
            print(f"    {r_['feature']:24s} {r_['rho']:+.2f}  p {r_['p']:.4f}  q {r_['q']:.3f}  | partial {r_['partial_rho']:+.2f}  p {r_['partial_p']:.4f}  q {r_['partial_q']:.3f}"
                  f"  | noGLM {r_.get('rho_no_glm', float('nan')):+.2f}  | acc {r_['mean_acc']:.2f}")
