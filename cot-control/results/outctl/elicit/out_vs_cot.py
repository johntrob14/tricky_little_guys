"""No-CoT output control (plain prompt) vs elicited CoT control, same models, items and constraint types (72 held-out items, 8 types).
Rows from dash/rows_long.jsonl (export_compare.py): measure out_nocot (gated strict on the response, CoT off) and cot_elicit (strict on the CoT).
  1. Type level: per model, pass rate per constraint type in each setting; Spearman across the 8 types within each model; pooled over
     model x type cells, also after removing each model's mean (within-model profile) and each type's mean (within-type).
  2. Item level: per model, items with both settings; P(CoT pass | output pass) vs P(CoT pass | output fail) (sample-averaged per item), and
     the pooled item-level correlation within model (Pearson of item pass rates, model-demeaned).
  3. Bound: share of model x type cells where elicited CoT > output (+ a 1-sample margin); null = shuffle the CoT type rates within each model
     (keeps every model's average and each setting's spread) 10k times; one-sided p = share of shuffles with as few violations.
Groups: all models, base models only (organisms share parents, so they are not independent).   python3 elicit/out_vs_cot.py -> dash/out_vs_cot.json"""
import collections, json, math, os, random
O = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(f"{O}/dash/rows_long.jsonl")]
both = sorted({r["model"] for r in R if r["measure"] == "out_nocot"} & {r["model"] for r in R if r["measure"] == "cot_elicit"})
GROUP = {r["model"]: r["group"] for r in R}
cell = collections.defaultdict(list); item = collections.defaultdict(list)
for r in R:
    if r["model"] in both and r["measure"] in ("out_nocot", "cot_elicit"):
        cell[(r["model"], r["constraint"], r["measure"])].append(r["ok"]); item[(r["model"], r["item"], r["measure"])].append(r["ok"])
TYPES = sorted({k[1] for k in cell})
mean = lambda v: sum(v) / len(v)
def rank(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]: j += 1
        for q in range(i, j + 1): r[o[q]] = (i + j) / 2
        i = j + 1
    return r
def pear(a, b):
    n = len(a); ma, mb = mean(a), mean(b); sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa and sb else float("nan")
spear = lambda a, b: pear(rank(a), rank(b))
rng = random.Random(0); res = {}
for gname, ms in [("all", both), ("base", [m for m in both if GROUP[m] == "Base model"])]:
    T = {m: {t: (100 * mean(cell[(m, t, "out_nocot")]), 100 * mean(cell[(m, t, "cot_elicit")])) for t in TYPES if cell[(m, t, "out_nocot")] and cell[(m, t, "cot_elicit")]} for m in ms}
    # 1. type level
    within = {m: spear([v[0] for v in T[m].values()], [v[1] for v in T[m].values()]) for m in ms}
    wv = [x for x in within.values() if not math.isnan(x)]
    cells = [(m, t, *T[m][t]) for m in ms for t in T[m]]
    mo = {m: (mean([v[0] for v in T[m].values()]), mean([v[1] for v in T[m].values()])) for m in ms}
    to = {t: (mean([c[2] for c in cells if c[1] == t]), mean([c[3] for c in cells if c[1] == t])) for t in TYPES}
    pooled = pear([c[2] for c in cells], [c[3] for c in cells])
    within_model = pear([c[2] - mo[c[0]][0] for c in cells], [c[3] - mo[c[0]][1] for c in cells])
    within_type = pear([c[2] - to[c[1]][0] for c in cells], [c[3] - to[c[1]][1] for c in cells])
    # null for within-model profile correlation: shuffle type labels of the CoT side within each model
    def wm(cs): return pear([c[2] - mo[c[0]][0] for c in cs], [c[3] - mo[c[0]][1] for c in cs])
    null = []
    for _ in range(5000):
        cs = []
        for m in ms:
            ts = list(T[m]); cv = [T[m][t][1] for t in ts]; rng.shuffle(cv); cs += [(m, t, T[m][t][0], c) for t, c in zip(ts, cv)]
        null.append(wm(cs))
    p_profile = (sum(x >= within_model for x in null) + 1) / (len(null) + 1)
    # 2. item level
    I = []
    for m in ms:
        for it in {k[1] for k in item if k[0] == m}:
            a, b = item.get((m, it, "out_nocot")), item.get((m, it, "cot_elicit"))
            if a and b: I.append((m, it, mean(a), mean(b)))
    io = {m: (mean([x[2] for x in I if x[0] == m]), mean([x[3] for x in I if x[0] == m])) for m in ms}
    item_r = pear([x[2] - io[x[0]][0] for x in I], [x[3] - io[x[0]][1] for x in I])
    # item level within model AND type (removes type difficulty): demean by model x type
    IT = collections.defaultdict(list)
    for x in I: IT[(x[0], x[1].split("|")[3] if x[1].count("|") >= 3 else None)].append(x)
    def ctype(itkey): return itkey.split("|")[3]
    mt = collections.defaultdict(list)
    for x in I: mt[(x[0], ctype(x[1]))].append(x)
    mtm = {k: (mean([x[2] for x in v]), mean([x[3] for x in v])) for k, v in mt.items()}
    item_r_mt = pear([x[2] - mtm[(x[0], ctype(x[1]))][0] for x in I], [x[3] - mtm[(x[0], ctype(x[1]))][1] for x in I])
    p_cond = (sum(x[3] for x in I if x[2] >= .5) / max(1, sum(1 for x in I if x[2] >= .5)), sum(x[3] for x in I if x[2] < .5) / max(1, sum(1 for x in I if x[2] < .5)))
    # 3. bound
    margin = 0.0
    viol = sum(c[3] > c[2] + margin for c in cells); nv = []
    for _ in range(10000):
        k = 0
        for m in ms:
            ts = list(T[m]); cv = [T[m][t][1] for t in ts]; rng.shuffle(cv); k += sum(c > T[m][t][0] + margin for t, c in zip(ts, cv))
        nv.append(k)
    p_bound = (sum(x <= viol for x in nv) + 1) / (len(nv) + 1)
    agg_viol = sum(mo[m][1] > mo[m][0] for m in ms)
    by_type = {t: dict(out=round(to[t][0], 1), cot=round(to[t][1], 1), cells_cot_above=sum(1 for c in cells if c[1] == t and c[3] > c[2]), n=sum(1 for c in cells if c[1] == t)) for t in TYPES}
    res[gname] = dict(models=ms, n_models=len(ms), n_cells=len(cells), n_items=len(I),
                      type_spearman_within_model_median=round(sorted(wv)[len(wv) // 2], 2), type_spearman_within_model=dict((m, round(v, 2)) for m, v in within.items()),
                      cells_pearson_pooled=round(pooled, 2), cells_pearson_within_model=round(within_model, 2), p_within_model_profile=round(p_profile, 4),
                      cells_pearson_within_type=round(within_type, 2), item_pearson_within_model=round(item_r, 2), item_pearson_within_model_type=round(item_r_mt, 2),
                      p_cot_pass_given_out_pass=round(p_cond[0], 3), p_cot_pass_given_out_fail=round(p_cond[1], 3),
                      cells_cot_above_out=viol, cells_cot_above_out_null_mean=round(mean(nv), 1), p_bound=round(p_bound, 4), models_cot_above_out_on_average=agg_viol,
                      by_type=by_type, cells=[dict(model=c[0], type=c[1], out=round(c[2], 1), cot=round(c[3], 1)) for c in cells])
    x = res[gname]
    print(f"== {gname}: {x['n_models']} models, {x['n_cells']} model x type cells, {x['n_items']} model x item pairs")
    print(f"  type profile: within-model Spearman median {x['type_spearman_within_model_median']}; cells Pearson pooled {x['cells_pearson_pooled']}, "
          f"within-model {x['cells_pearson_within_model']} (shuffle p {x['p_within_model_profile']}), within-type {x['cells_pearson_within_type']}")
    print(f"  items: Pearson within model {x['item_pearson_within_model']}, within model x type {x['item_pearson_within_model_type']}; "
          f"P(CoT pass | output pass) {x['p_cot_pass_given_out_pass']} vs | output fail {x['p_cot_pass_given_out_fail']}")
    print(f"  bound: CoT above output in {viol}/{x['n_cells']} cells (shuffled-types null {x['cells_cot_above_out_null_mean']}, p {x['p_bound']}); models above on average {agg_viol}/{x['n_models']}")
    for t, v in by_type.items(): print(f"    {t:28s} out {v['out']:5.1f}  cot {v['cot']:5.1f}  cells cot>out {v['cells_cot_above']}/{v['n']}")
json.dump(res, open(f"{O}/dash/out_vs_cot.json", "w"), indent=1)

# 4. is the gap length? logistic on both settings' rows: intercept per (model, setting), per-type intercept and per-type length slope
#    (separate slopes per setting, since output and CoT text may scale differently), evaluated on each model's pooled rows of both settings.
#    Also: violations larger than sampling noise (cot - out > 2 x the binomial SE of the difference).
import numpy as np
def fitlog(rows, groups, lam=1.0, iters=500):
    G = {g: i for i, g in enumerate(groups)}; K = {k: i for i, k in enumerate(sorted({(r["measure"], r["constraint"]) for r in rows}))}; nG, nK = len(G), len(K)
    X = np.zeros((len(rows), nG + 2 * nK)); y = np.array([r["ok"] for r in rows], float)
    for i, r in enumerate(rows):
        k = K[(r["measure"], r["constraint"])]; X[i, G[(r["model"], r["measure"])]] = 1; X[i, nG + k] = 1; X[i, nG + nK + k] = math.log2(r["words"] + 1)
    w = np.zeros(X.shape[1]); P = lam * np.eye(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w))); W = p * (1 - p)
        step = np.clip(np.linalg.solve(X.T @ (X * W[:, None]) + P, X.T @ (y - p) - P @ w), -1, 1); w += step
        if np.abs(step).max() < 1e-7: break
    return w, G, K, nG, nK
rows = [r for r in R if r["model"] in both and r["measure"] in ("out_nocot", "cot_elicit") and r["words"] is not None]
w, G, K, nG, nK = fitlog(rows, sorted({(r["model"], r["measure"]) for r in rows}))
def p_at(m, meas, rs):  # constraint-balanced predicted pass for setting meas, evaluated at rows rs (their type and length; slope of meas)
    by = collections.defaultdict(list)
    for r in rs:
        k = K[(meas, r["constraint"])]; by[r["constraint"]].append(1 / (1 + math.exp(-(w[G[(m, meas)]] + w[nG + k] + w[nG + nK + k] * math.log2(r["words"] + 1)))))
    return 100 * sum(mean(v) for v in by.values()) / len(by)
gap = []
for m in both:
    mine = [r for r in rows if r["model"] == m]; O_ = [r for r in mine if r["measure"] == "out_nocot"]; C_ = [r for r in mine if r["measure"] == "cot_elicit"]
    raw = p_at(m, "out_nocot", O_) - p_at(m, "cot_elicit", C_)
    at_cot_len = p_at(m, "out_nocot", C_) - p_at(m, "cot_elicit", C_)   # output scored at the CoT's lengths
    at_out_len = p_at(m, "out_nocot", O_) - p_at(m, "cot_elicit", O_)   # CoT scored at the output's lengths
    gap.append(dict(model=m, group=GROUP[m], out_words=sorted(r["words"] for r in O_)[len(O_) // 2], cot_words=sorted(r["words"] for r in C_)[len(C_) // 2],
                    gap_raw_fit=round(raw, 1), gap_at_cot_length=round(at_cot_len, 1), gap_at_output_length=round(at_out_len, 1)))
print("\noutput minus elicited CoT, model fit: raw / both at the CoT's lengths / both at the output's lengths")
for g in sorted(gap, key=lambda g: -g["gap_raw_fit"]):
    print(f"  {g['model']:38s} words out {g['out_words']:4d} cot {g['cot_words']:4d}   gap {g['gap_raw_fit']:+6.1f}  at CoT len {g['gap_at_cot_length']:+6.1f}  at output len {g['gap_at_output_length']:+6.1f}")
neg = lambda k: sum(g[k] < 0 for g in gap)
print(f"  models with CoT above output: raw {neg('gap_raw_fit')}/{len(gap)}, at CoT length {neg('gap_at_cot_length')}, at output length {neg('gap_at_output_length')}")
big = []
for c in res["all"]["cells"]:
    no, nc = len(cell[(c["model"], c["type"], "out_nocot")]), len(cell[(c["model"], c["type"], "cot_elicit")])
    po, pc = c["out"] / 100, c["cot"] / 100; se = 100 * math.sqrt(max(po * (1 - po), 1 / no) / no + max(pc * (1 - pc), 1 / nc) / nc)
    if c["cot"] - c["out"] > 2 * se: big.append(dict(c, se=round(se, 1)))
print(f"  cells where CoT exceeds output by > 2 SE: {len(big)}/160 (expected ~4 by chance at 2.5% one-sided)")
for b in big: print("   ", b)
res["gap_by_length"] = gap; res["cells_cot_above_out_beyond_noise"] = big
json.dump(res, open(f"{O}/dash/out_vs_cot.json", "w"), indent=1)
