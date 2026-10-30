"""CSV exports for the per-model CoT vs output control dashboard (base gpt-oss-120b and manytypes1b s150).
Arms: CoT = constraint on the analysis channel, medium effort (side cot); Output = constraint on the response, analysis channel forced
empty (side expl). Items: 108 Astra (more/) + 432 new (big/). Length = words of the constrained text (CoT for CoT, response for Output).
Run from results/outctl:  python3 export_dash_v4.py  -> dash_v4/{summary,length_bins,length_slopes,task_scatter}.csv"""
import csv, json, math, os, collections, statistics as st
# run id -> (label, kind); order is the page's order. A model is exported only when all four graded files exist.
MODELS = {"base": ("gpt-oss-120b", "base"),
          "mt1b-s150": ("manytypes1b s150", "CoT-control RL, many types"), "grpo3": ("grpo3 s100", "CoT-control RL"),
          "st1-ignore2": ("st1-ignore2 s60", "CoT-control RL, ignore_question"), "st1-ignore3": ("st1-ignore3 s60", "CoT-control RL, ignore_question"),
          "st1-third": ("st1-third s60", "CoT-control RL, third person"), "gsft1": ("gsft1 ep2", "CoT-control SFT"),
          "len2": ("len2 s60", "CoT length RL"), "efflen1": ("efflen1 s100", "CoT length RL, effort-cued"),
          "eff1": ("eff1 s40", "CoT effort RL"), "eff2": ("eff2 s40", "CoT effort RL"), "effm1": ("effm1 s30", "CoT effort RL"), "effm2": ("effm2 s50", "CoT effort RL"),
          "rgrpo1-s10": ("rgrpo1 s10", "response-control RL"),
          "rlen1-s10": ("rlen1 s10", "response length RL"), "rlen1-s20": ("rlen1 s20", "response length RL"), "rlen1-s30": ("rlen1 s30", "response length RL")}
ARMS = {"cot": "CoT", "expl": "Output"}
EDGES = [25, 50, 100, 200, 400, 800, 1600, 10**9]
LAB = ["1-25", "26-50", "51-100", "101-200", "201-400", "401-800", "801-1600", "1601+"]
MID = [12.5, 37.5, 75, 150, 300, 600, 1200, 2400]

def load(side, m):
    rows = []
    for d in ("more", "big"):
        for x in json.load(open(f"{d}/{side}/{m}/graded.json")):
            rows.append(dict(model=m, arm=ARMS[side], item=f"{d}|{x['key']}", dataset=x["dataset"], c=x["constraint"],
                             ok=int(x["strict"]), w=int(x["words"]), correct=x["correct"] in (True, "True"), trunc=x.get("stop") == "length"))
    return rows

def wilson(k, n, z=1.96):
    if not n: return None, None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

def logit_slope(rows):
    """Logistic regression of pass on log2(words) with dataset fixed effects (one constraint's rows); Newton steps, no numpy."""
    ds = sorted({r["dataset"] for r in rows})
    X = [[math.log2(max(r["w"], 1))] + [float(r["dataset"] == d) for d in ds] for r in rows]; y = [r["ok"] for r in rows]
    k = len(X[0]); w = [0.0] * k
    if sum(y) in (0, len(y)): return None, None
    for _ in range(50):
        g = [0.0] * k; H = [[1e-6 * (i == j) for j in range(k)] for i in range(k)]
        for xi, yi in zip(X, y):
            p = 1 / (1 + math.exp(-max(-30, min(30, sum(a * b for a, b in zip(w, xi))))))
            for i in range(k):
                g[i] += (yi - p) * xi[i]
                for j in range(k): H[i][j] += p * (1 - p) * xi[i] * xi[j]
        Hi = inv(H); step = [sum(Hi[i][j] * g[j] for j in range(k)) for i in range(k)]
        w = [a + b for a, b in zip(w, step)]
        if max(abs(s) for s in step) < 1e-8: break
    if abs(w[0]) > 10: return None, None  # separation
    return w[0], math.sqrt(Hi[0][0])

def inv(A):
    n = len(A); M = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c])); M[c], M[p] = M[p], M[c]
        pv = M[c][c]; M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c: f = M[r][c]; M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]

def bin_of(w): return next(i for i, e in enumerate(EDGES) if w <= e)

def main():
    os.makedirs("dash_v4", exist_ok=True)
    have = [m for m in MODELS if all(os.path.exists(f"{d}/{s}/{m}/graded.json") for d in ("more", "big") for s in ARMS)]
    print("exporting", have, "skipping", [m for m in MODELS if m not in have])
    allrows = [r for m in have for side in ARMS for r in load(side, m)]
    summ, bins, slopes, scat = [], [], [], []
    for order, m in enumerate(have):
        mname, kind = MODELS[m]
        for arm in ARMS.values():
            rs = [r for r in allrows if r["model"] == m and r["arm"] == arm]
            ni = [r for r in rs if r["c"] != "ignore_question"]; iq = [r for r in rs if r["c"] == "ignore_question"]
            summ.append(dict(model=mname, kind=kind, order=order, arm=arm, samples=len(rs), items=len({r["item"] for r in rs}),
                             pass_excl_ignore=round(sum(r["ok"] for r in ni) / len(ni), 4), pass_ignore_question=round(sum(r["ok"] for r in iq) / len(iq), 4),
                             accuracy=round(sum(r["correct"] for r in rs) / len(rs), 4), median_words=st.median(r["w"] for r in rs), truncated=sum(r["trunc"] for r in rs)))
            for c in sorted({r["c"] for r in rs}):
                cr = [r for r in rs if r["c"] == c]
                for i in range(len(EDGES)):
                    b = [r["ok"] for r in cr if bin_of(r["w"]) == i]
                    if not b: continue
                    lo, hi = wilson(sum(b), len(b))
                    bins.append(dict(model=mname, arm=arm, constraint=c, bin=LAB[i], bin_order=i, bin_mid_words=MID[i], n=len(b), passes=sum(b),
                                     rate=round(sum(b) / len(b), 4), lo95=round(lo, 4), hi95=round(hi, 4)))
                s, se = logit_slope(cr)
                slopes.append(dict(model=mname, arm=arm, constraint=c, n=len(cr), pass_rate=round(sum(r["ok"] for r in cr) / len(cr), 4),
                                   median_words=st.median(r["w"] for r in cr), logit_per_doubling=None if s is None else round(s, 3), se=None if se is None else round(se, 3)))
        for c in sorted({r["c"] for r in allrows}):
            for d in sorted({r["dataset"] for r in allrows}):
                g = {arm: [r["ok"] for r in allrows if r["model"] == m and r["arm"] == arm and r["c"] == c and r["dataset"] == d] for arm in ARMS.values()}
                scat.append(dict(model=mname, constraint=c, dataset=d, task=f"{c} / {d}", cot_n=len(g["CoT"]), cot_rate=round(st.mean(g["CoT"]), 4),
                                 output_n=len(g["Output"]), output_rate=round(st.mean(g["Output"]), 4)))
    for name, rows in (("summary", summ), ("length_bins", bins), ("length_slopes", slopes), ("task_scatter", scat)):
        with open(f"dash_v4/{name}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        print(name, len(rows))

if __name__ == "__main__":
    main()
