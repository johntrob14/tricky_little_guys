"""CSV exports for the Output vs CoT Control dashboard, from big/rows_v3.json (analyze_v3.py). Run from results/outctl."""
import csv, json, random, collections, statistics as st
D = json.load(open("big/rows_v3.json"))
KIND = {"base": "base", "grpo3": "CoT-control RL", "st1-ignore2": "CoT-control RL", "st1-ignore3": "CoT-control RL", "gsft1": "CoT-control SFT",
        "len2": "CoT length RL", "efflen1": "CoT length RL", "st1-third": "CoT-control RL (third person only)"}
EDGES = [5, 20, 50, 100, 200, 500, 10**9]; LAB = ["0-5", "6-20", "21-50", "51-100", "101-200", "201-500", "500+"]
def b(w): return next(i for i, e in enumerate(EDGES) if w <= e)
def gap(groups_a, groups_b):
    num = den = 0
    for k in set(groups_a) | set(groups_b):
        a, c = groups_a.get(k, []), groups_b.get(k, [])
        if a and c: w = min(len(a), len(c)); num += w * (sum(c) / len(c) - sum(a) / len(a)); den += w
    return (100 * num / den if den else None), den
def strata(rows, arms, cons=None):
    s = collections.defaultdict(list)
    for r in rows:
        if r["arm"] in arms and (cons is None or r["c"] == cons): s[(r["c"], b(r["w"]))].append(r["ok"])
    return s
def keyed(rows): 
    d = collections.defaultdict(list)
    for r in rows: d[tuple(r["item"]) + (r["c"],)].append(r)
    return d
def boot(f, keys, n=1000):
    rng = random.Random(0); v = []
    for _ in range(n):
        x = f(rng.choices(keys, k=len(keys)))
        if x is not None: v.append(x)
    v.sort(); return v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]
COT, EXP = {"cot_med", "cot_low"}, {"expl"}
rate = lambda rows, arms: 100 * sum(r["ok"] for r in rows if r["arm"] in arms) / sum(r["arm"] in arms for r in rows)
head, vsb = [], []
bk = keyed(D["base"])
for m, rows in D.items():
    km = keyed(rows); keys = sorted(km)
    g, w = gap(strata(rows, COT), strata(rows, EXP))
    lo, hi = boot(lambda ks: gap(strata([r for k in ks for r in km[k]], COT), strata([r for k in ks for r in km[k]], EXP))[0], keys)
    head.append(dict(model=m, kind=KIND[m], cot_medium=round(rate(rows, {"cot_med"}), 1), cot_low=round(rate(rows, {"cot_low"}), 1), cot=round(rate(rows, COT), 1),
                     explained_output=round(rate(rows, EXP), 1), gap=round(g, 1), gap_lo=round(lo, 1), gap_hi=round(hi, 1), gap_weight=w,
                     median_words_cot=st.median(r["w"] for r in rows if r["arm"] in COT), median_words_output=st.median(r["w"] for r in rows if r["arm"] in EXP),
                     n_cot=sum(r["arm"] in COT for r in rows), n_output=sum(r["arm"] in EXP for r in rows)))
    acc = 100 * sum(r["cor"] for r in rows if r["arm"] in EXP) / sum(r["arm"] in EXP for r in rows)
    if m == "base": vsb.append(dict(model=m, kind=KIND[m], explained_output=round(rate(rows, EXP), 1), vs_base=0.0, vs_base_lo=0.0, vs_base_hi=0.0, output_accuracy=round(acc, 1))); continue
    f = lambda ks: gap(strata([r for k in ks for r in bk[k]], EXP), strata([r for k in ks for r in km[k]], EXP))[0]
    g2, _ = gap(strata(D["base"], EXP), strata(rows, EXP)); lo2, hi2 = boot(f, sorted(bk))
    vsb.append(dict(model=m, kind=KIND[m], explained_output=round(rate(rows, EXP), 1), vs_base=round(g2, 1), vs_base_lo=round(lo2, 1), vs_base_hi=round(hi2, 1), output_accuracy=round(acc, 1)))
bins, cons = [], []
for m, rows in D.items():
    for arm, arms in (("CoT", COT), ("Explained output", EXP)):
        c = [[0, 0] for _ in LAB]
        for r in rows:
            if r["arm"] in arms: k = c[b(r["w"])]; k[0] += r["ok"]; k[1] += 1
        for i, (p, n) in enumerate(c):
            bins.append(dict(model=m, arm=arm, bin=LAB[i], bin_order=i, passes=p, n=n, rate=round(100 * p / n, 1) if n else ""))
    for t in sorted({r["c"] for r in rows}):
        g, w = gap(strata(rows, COT, t), strata(rows, EXP, t))
        cons.append(dict(model=m, constraint=t, cot=round(rate([r for r in rows if r["c"] == t], COT), 1), explained_output=round(rate([r for r in rows if r["c"] == t], EXP), 1),
                         gap=round(g, 1) if g is not None else "", gap_weight=w))
for name, rows in (("headline", head), ("vs_base", vsb), ("length_bins", bins), ("by_constraint", cons)):
    with open(f"dash/{name}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(name, len(rows))
