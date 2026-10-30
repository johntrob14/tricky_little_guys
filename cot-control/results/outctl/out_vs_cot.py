"""Output control (CoT off, explained-output arm 'expl') vs CoT control ('cot' side, medium) per model, on the 540 CoT-Control items.
Gated pass = strict pass AND the attempt judge (as analyze_v3.py); ignore_question excluded (its output version differs). Reports raw gated
pass per arm, median words, the output - CoT gap matched on constraint x length bin (weights = min count per cell), per-constraint pass per arm,
and output-arm length curves + Kitagawa split of each model's OUTPUT gain vs base's output arm.
python3 out_vs_cot.py MODEL [...]   -> out_vs_cot.json"""
import json, os, sys, collections, random
O = os.path.dirname(os.path.abspath(__file__))
WB = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 10**9)]; LAB = ["<50", "50-100", "100-200", "200-400", "400+"]
def rows(m, arm):
    out = []
    for d in ("more", "big"):
        p = f"{O}/{d}/{arm}/{m}/graded.json"
        if not os.path.exists(p): continue
        for g in json.load(open(p)):
            if g["constraint"] == "ignore_question": continue
            st = str(g.get("strict")) in ("1", "True"); att = g.get("attempt")
            out.append(dict(item=g["key"].rsplit("|", 1)[0] if g["key"].startswith("astra|") else g["key"], c=g["constraint"], w=int(g.get("words") or 0), ok=bool(st and att is not False and att is not None)))
    return out
bi = lambda w: next(i for i, (a, z) in enumerate(WB) if a <= w < z)
def matched_gap(A, B):  # A - B matched on constraint x length bin
    ca, cb = collections.defaultdict(list), collections.defaultdict(list)
    for r in A: ca[(r["c"], bi(r["w"]))].append(r["ok"])
    for r in B: cb[(r["c"], bi(r["w"]))].append(r["ok"])
    num = den = 0
    for k in set(ca) & set(cb):
        w = min(len(ca[k]), len(cb[k])); num += w * (sum(ca[k]) / len(ca[k]) - sum(cb[k]) / len(cb[k])); den += w
    return 100 * num / den if den else None, den
def curve(R):
    n = collections.Counter(bi(r["w"]) for r in R); s = collections.Counter(bi(r["w"]) for r in R if r["ok"])
    return [n[i] / len(R) for i in range(len(WB))], [s[i] / n[i] if n[i] else None for i in range(len(WB))], [n[i] for i in range(len(WB))]
BO = rows("base", "expl"); pb, cb, _ = curve(BO)
res = []
print("| model | CoT gated pass | output gated pass | median words CoT / output | output - CoT matched (weight) | output gain vs base output = length + control |")
for m in sys.argv[1:]:
    C, X = rows(m, "cot"), rows(m, "expl")
    if not C or not X: print(m, "missing", len(C), len(X)); continue
    gap, wt = matched_gap(X, C)
    pm, cm, nm = curve(X); cmf = [cm[i] if cm[i] is not None else (cb[i] or 0) for i in range(len(WB))]
    lp = sum((pm[i] - pb[i]) * (cb[i] or 0) for i in range(len(WB))); cp = sum(pm[i] * (cmf[i] - (cb[i] or 0)) for i in range(len(WB)))
    per = {}
    for c in sorted({r["c"] for r in C}):
        a = [r["ok"] for r in C if r["c"] == c]; b_ = [r["ok"] for r in X if r["c"] == c]
        per[c] = dict(cot=round(100 * sum(a) / len(a), 1) if a else None, out=round(100 * sum(b_) / len(b_), 1) if b_ else None)
    med = lambda R: sorted(r["w"] for r in R)[len(R) // 2]
    o = dict(model=m, cot=round(100 * sum(r["ok"] for r in C) / len(C), 1), out=round(100 * sum(r["ok"] for r in X) / len(X), 1), med_cot=med(C), med_out=med(X), gap=round(gap, 1), gap_weight=wt,
             out_length_part=round(100 * lp, 1), out_control_part=round(100 * cp, 1), out_curve=[None if x is None else round(100 * x, 1) for x in cm], out_share=[round(x, 3) for x in pm], out_n=nm, per=per, n_cot=len(C), n_out=len(X))
    res.append(o); print(f"| {m} | {o['cot']} | {o['out']} | {o['med_cot']} / {o['med_out']} | {o['gap']:+} ({wt}) | {o['out']-res[0]['out'] if res else 0:+.1f} = {o['out_length_part']:+} + {o['out_control_part']:+} |")
json.dump(dict(bins=LAB, base_out_curve=[None if x is None else round(100 * x, 1) for x in cb], models=res), open(f"{O}/out_vs_cot.json", "w"), indent=1)
