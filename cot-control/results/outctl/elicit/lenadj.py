"""How much of each result is length? Row-level data from export_compare.py (dash/rows_long.jsonl: pass and words of the graded text per sample,
72 held-out items, 8 types). One logistic model per measure family, L2-penalised (lam 1; base rates of 0 would otherwise diverge):
  logit P(pass) = a[model, condition] + b[constraint] + s[constraint] * log2(words + 1)
The length slopes s are shared across models and conditions (identified from within-model, within-condition variation in length).
  1. Elicitation: CoT rows, condition = as written / elicited. Gain at matched length = the two intercepts evaluated on the same rows
     (the model's elicited rows, i.e. what the as-written prompt would score at elicited lengths); length part = raw gain - matched gain.
  2. Cross-model standardisation: per measure, each model's predicted pass on a common reference set (every row of that measure, all models
     pooled: same constraints and lengths for everyone) = control at a common length.
  3. NCRI: Spearman / Pearson of NCRI with raw and with standardised scores (ncri_corr.py's model map; permutation p, 100k).
95% CIs by bootstrap over items (refit each draw; 300 draws, seed 0) for the elicitation gains.
python3 elicit/lenadj.py  -> elicit/dash/lenadj_gain.json, lenadj_std.json, lenadj_ncri.json"""
import collections, json, math, os, random
import numpy as np
O = os.path.dirname(os.path.abspath(__file__))
R = [r for r in map(json.loads, open(f"{O}/dash/rows_long.jsonl")) if r["words"] is not None]
TYPES = sorted({r["constraint"] for r in R})

def fit(rows, groups, lam=1.0, iters=500):
    """IRLS for logit = a[group] + b[type] + s[type] * L. Returns params dict."""
    G = {g: i for i, g in enumerate(groups)}; T = {t: i for i, t in enumerate(TYPES)}; nG, nT = len(G), len(T)
    X = np.zeros((len(rows), nG + 2 * nT)); y = np.array([r["ok"] for r in rows], float)
    for i, r in enumerate(rows):
        L = math.log2(r["words"] + 1)
        X[i, G[r["g"]]] = 1; X[i, nG + T[r["constraint"]]] = 1; X[i, nG + nT + T[r["constraint"]]] = L
    w = np.zeros(X.shape[1]); P = lam * np.eye(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w))); W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + P; g = X.T @ (y - p) - P @ w
        step = np.clip(np.linalg.solve(H, g), -1, 1); w += step
        if np.abs(step).max() < 1e-7: break
    return dict(a={g: w[i] for g, i in G.items()}, b={t: w[nG + i] for t, i in T.items()}, s={t: w[nG + nT + i] for t, i in T.items()})
def pred(par, g, rows):
    return float(np.mean([1 / (1 + math.exp(-(par["a"][g] + par["b"][r["constraint"]] + par["s"][r["constraint"]] * math.log2(r["words"] + 1)))) for r in rows]))
def bal(rows):  # constraint-balanced observed pass, as the dashboard
    by = collections.defaultdict(list)
    for r in rows: by[r["constraint"]].append(r["ok"])
    return 100 * sum(sum(v) / len(v) for v in by.values()) / len(by)
def balpred(par, g, rows):  # constraint-balanced predicted pass over rows
    by = collections.defaultdict(list)
    for r in rows: by[r["constraint"]].append(r)
    return 100 * sum(pred(par, g, v) for v in by.values()) / len(by)
def med(xs): xs = sorted(xs); return xs[len(xs) // 2]

# 1. elicitation, matched length
cot = [dict(r, g=(r["model"], r["measure"])) for r in R if r["measure"] in ("cot_base", "cot_elicit")]
pair = sorted({r["model"] for r in cot if r["measure"] == "cot_base"} & {r["model"] for r in cot if r["measure"] == "cot_elicit"})
def gains(rows):
    par = fit(rows, sorted({r["g"] for r in rows}))
    out = {}
    for m in pair:
        B = [r for r in rows if r["g"] == (m, "cot_base")]; E = [r for r in rows if r["g"] == (m, "cot_elicit")]
        if not B or not E: continue
        out[m] = dict(raw=bal(E) - bal(B), matched_at_elicited=balpred(par, (m, "cot_elicit"), E) - balpred(par, (m, "cot_base"), E),
                      matched_at_base=balpred(par, (m, "cot_elicit"), B) - balpred(par, (m, "cot_base"), B))
    return out, par
G0, par0 = gains(cot)
items = sorted({r["item"] for r in cot}); rng = random.Random(0); boot = collections.defaultdict(list)
for _ in range(300):
    pick = collections.Counter(rng.choice(items) for _ in items)
    rows = [r for r in cot for _ in range(pick[r["item"]])]
    for m, d in gains(rows)[0].items(): boot[m].append(d["matched_at_elicited"])
words = {(m, k): med([r["words"] for r in cot if r["g"] == (m, k)]) for m in pair for k in ("cot_base", "cot_elicit")}
gain = []
print("length slopes per type (log-odds per doubling of CoT words):", {t: round(v, 2) for t, v in par0["s"].items()})
print(f"{'model':38s} {'words base->elicited':>22s} {'raw gain':>9s} {'matched (at elicited len)':>28s} {'length part':>12s}")
for m in pair:
    d = G0[m]; b = sorted(boot[m]); lo, hi = b[int(.025 * len(b))], b[int(.975 * len(b))]
    gain.append(dict(model=m, words_base=words[(m, "cot_base")], words_elicit=words[(m, "cot_elicit")], raw_gain=round(d["raw"], 1),
                     matched_gain=round(d["matched_at_elicited"], 1), matched_lo=round(lo, 1), matched_hi=round(hi, 1),
                     matched_gain_at_base_len=round(d["matched_at_base"], 1), length_part=round(d["raw"] - d["matched_at_elicited"], 1)))
    x = gain[-1]; print(f"{m:38s} {x['words_base']:>9d} -> {x['words_elicit']:<9d} {x['raw_gain']:+9.1f} {x['matched_gain']:+12.1f} [{lo:+.1f}, {hi:+.1f}]   {x['length_part']:+10.1f}")
json.dump(gain, open(f"{O}/dash/lenadj_gain.json", "w"), indent=1)

# 2. standardise each measure across models to a common length distribution
std = []
for k in ("out_nocot", "out_cot", "cot_base", "cot_elicit"):
    rows = [dict(r, g=r["model"]) for r in R if r["measure"] == k]
    par = fit(rows, sorted({r["g"] for r in rows}))
    for m in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == m]
        std.append(dict(model=m, group=mine[0]["group"], measure=k, raw=round(bal(mine), 1), standardised=round(balpred(par, m, rows), 1),
                        median_words=med([r["words"] for r in mine])))
json.dump(std, open(f"{O}/dash/lenadj_std.json", "w"), indent=1)

# 3. NCRI with raw vs standardised scores
import csv
NC = {r["slug"]: float(r["ncri15_2"]) for r in csv.DictReader(open("/home/john/code/neel-astra/raw/ncri_raw_data/models_ncri15_2.csv"))}
SLUG = eval(open(f"{O}/ncri_corr.py").read().split("SLUG = ", 1)[1].split("\n# not in")[0])  # ncri_corr.py's model -> NCRI slug map
def rank(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]: j += 1
        for q in range(i, j + 1): r[o[q]] = (i + j) / 2
        i = j + 1
    return r
def pear(a, b):
    n = len(a); ma, mb = sum(a) / n, sum(b) / n; sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)
spear = lambda a, b: pear(rank(a), rank(b))
def perm_p(a, b, rng, B=100000):
    r0, bb, k = abs(spear(a, b)), b[:], 0
    for _ in range(B): rng.shuffle(bb); k += abs(spear(a, bb)) >= r0 - 1e-12
    return (k + 1) / (B + 1)
rng = random.Random(0); nc = []
gm = {x["model"]: x for x in gain}
for k, lab in [("out_nocot", "Output control, CoT off"), ("out_cot", "Output control, CoT on"), ("cot_base", "CoT control, as written"),
               ("cot_elicit", "CoT control, elicited"), ("gain", "Elicitation gain (points)")]:
    if k == "gain":
        ms = [m for m in gm if m in SLUG]; xs = [NC[SLUG[m]] for m in ms]; raw = [gm[m]["raw_gain"] for m in ms]; adj = [gm[m]["matched_gain"] for m in ms]
    else:
        S = {x["model"]: x for x in std if x["measure"] == k and x["group"] == "Base model" and x["model"] in SLUG}
        ms = sorted(S); xs = [NC[SLUG[m]] for m in ms]; raw = [S[m]["raw"] for m in ms]; adj = [S[m]["standardised"] for m in ms]
    if len(ms) < 4: continue
    row = dict(measure=lab, n=len(ms), spearman_raw=round(spear(xs, raw), 2), p_raw=round(perm_p(xs, raw, rng), 4),
               spearman_len=round(spear(xs, adj), 2), p_len=round(perm_p(xs, adj, rng), 4), pearson_raw=round(pear(xs, raw), 2), pearson_len=round(pear(xs, adj), 2))
    nc.append(row); print(row)
json.dump(nc, open(f"{O}/dash/lenadj_ncri.json", "w"), indent=1)

# 4. robustness: estimate the length slopes on one condition only (as-written rows or elicited rows), then hold them fixed and refit the
#    intercepts on all rows; report the length part of each elicitation gain under each slope source.
def fit_fixed_slopes(rows, groups, s, lam=1.0, iters=500):
    off = np.array([s[r["constraint"]] * math.log2(r["words"] + 1) for r in rows])
    G = {g: i for i, g in enumerate(groups)}; T = {t: i for i, t in enumerate(TYPES)}; nG, nT = len(G), len(T)
    X = np.zeros((len(rows), nG + nT)); y = np.array([r["ok"] for r in rows], float)
    for i, r in enumerate(rows): X[i, G[r["g"]]] = 1; X[i, nG + T[r["constraint"]]] = 1
    w = np.zeros(X.shape[1]); P = lam * np.eye(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + off))); W = p * (1 - p)
        step = np.clip(np.linalg.solve(X.T @ (X * W[:, None]) + P, X.T @ (y - p) - P @ w), -1, 1); w += step  # damped: plain Newton diverged here
        if np.abs(step).max() < 1e-7: break
    return dict(a={g: w[i] for g, i in G.items()}, b={t: w[nG + i] for t, i in T.items()}, s=s)
rob = []
for src in ("cot_base", "cot_elicit"):
    sub = [r for r in cot if r["measure"] == src]; s = fit(sub, sorted({r["g"] for r in sub}))["s"]
    par = fit_fixed_slopes(cot, sorted({r["g"] for r in cot}), s)
    print(f"slopes from {src} rows only:", {t: round(float(v), 2) for t, v in s.items()})
    for m in pair:
        E = [r for r in cot if r["g"] == (m, "cot_elicit")]; raw = gm[m]["raw_gain"]
        mg = balpred(par, (m, "cot_elicit"), E) - balpred(par, (m, "cot_base"), E)
        rob.append(dict(model=m, slope_source=src, matched_gain=round(mg, 1), length_part=round(raw - mg, 1)))
for m in pair:
    x = {r["slope_source"]: r for r in rob if r["model"] == m}
    print(f"{m:38s} length part: pooled {gm[m]['length_part']:+5.1f}  as-written slopes {x['cot_base']['length_part']:+5.1f}  elicited slopes {x['cot_elicit']['length_part']:+5.1f}")
json.dump(rob, open(f"{O}/dash/lenadj_robust.json", "w"), indent=1)
for x in gain:  # robustness columns on the per-model gain table
    for r in rob:
        if r["model"] == x["model"]: x["length_part_" + ("aswritten_slopes" if r["slope_source"] == "cot_base" else "elicited_slopes")] = r["length_part"]
json.dump(gain, open(f"{O}/dash/lenadj_gain.json", "w"), indent=1)
