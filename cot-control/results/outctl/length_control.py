"""Control vs CoT length on CoT-Control (cot side, 108 + 432 items): per-model curves of strict pass and partial compliance over log-length bins,
area under each curve on a fixed bin grid (equal weight per bin, a model's missing bin falls back to base's value), and the Kitagawa
decomposition of the raw strict gain vs base into a length-distribution part (valued at base's curve) and a control-at-length part.
Bootstrap (items) 95% CIs for the AUC differences and both decomposition parts.
Run from cot-control/rl:  uv run python ../results/outctl/length_control.py MODEL [...]"""
import json, os, random, sys, collections
sys.path.insert(0, ".")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "astra540.py")).read().split("bs, bc, bn = per_item")[0]
ns = {"__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "astra540.py")}; exec(src, ns)
import astra_continuous as AC
O = os.path.dirname(os.path.abspath(__file__))
BINS = [(0, 60), (60, 120), (120, 240), (240, 480), (480, 10**9)]
LAB = ["<60", "60-120", "120-240", "240-480", "480+"]
def rows(m):
    out = []
    for d in ("more", "big"):
        p = f"{O}/{d}/cot/{m}/graded.json"
        if not os.path.exists(p): continue
        for r in json.load(open(p)):
            k = r["key"]; ga = ns["GA"].get(k) or ns["GA"].get(ns["item_key"](r)) or {}
            st = str(r["strict"]) in ("1", "True")
            out.append(dict(item=(d, k), w=int(r["words"] or 0), s=float(st), c=AC.score(dict(constraint=r["constraint"], reasoning=r.get("body") or "", grader_args=ga), st)))
    return out
def b(w): return next(i for i, (a, z) in enumerate(BINS) if a <= w < z)
def stats(R, ref=None):
    n = collections.Counter(); s = collections.Counter(); c = collections.Counter()
    for r in R: i = b(r["w"]); n[i] += 1; s[i] += r["s"]; c[i] += r["c"]
    p = [n[i] / len(R) for i in range(len(BINS))]
    cs = [s[i] / n[i] if n[i] else (ref[1][i] if ref else 0.0) for i in range(len(BINS))]
    cc = [c[i] / n[i] if n[i] else (ref[2][i] if ref else 0.0) for i in range(len(BINS))]
    return p, cs, cc, n
def summary(R, B):
    pb, sb, cb, _ = stats(B); pm, sm, cm, _ = stats(R, (pb, sb, cb))
    auc_s = sum(sm) / len(BINS) - sum(sb) / len(BINS); auc_c = sum(cm) / len(BINS) - sum(cb) / len(BINS)
    length_part = sum((pm[i] - pb[i]) * sb[i] for i in range(len(BINS)))
    control_part = sum(pm[i] * (sm[i] - sb[i]) for i in range(len(BINS)))
    raw = sum(pm[i] * sm[i] for i in range(len(BINS))) - sum(pb[i] * sb[i] for i in range(len(BINS)))
    return dict(raw=raw, length=length_part, control=control_part, auc_s=auc_s, auc_c=auc_c)
def boot(R, B, n=400):
    bi = collections.defaultdict(list); mi = collections.defaultdict(list)
    for r in B: bi[r["item"][1]].append(r)
    for r in R: mi[r["item"][1]].append(r)
    keys = sorted(set(bi) & set(mi)); rng = random.Random(0); acc = collections.defaultdict(list)
    for _ in range(n):
        ks = [rng.choice(keys) for _ in keys]
        sm = summary([r for k in ks for r in mi[k]], [r for k in ks for r in bi[k]])
        for k, v in sm.items(): acc[k].append(v)
    return {k: (sorted(v)[int(.025 * n)], sorted(v)[int(.975 * n)]) for k, v in acc.items()}
B = rows("base"); pb, sb, cb, nb = stats(B)
print("bins (CoT words):", LAB)
print(f"{'base':16s} share {['%.2f' % x for x in pb]} strict {['%.1f' % (100 * x) for x in sb]} partial {['%.1f' % (100 * x) for x in cb]}")
OUTD, OUTC = [], []
for i, l in enumerate(LAB): OUTC.append(dict(model="base", bin=l, bin_i=i, share=round(pb[i], 3), strict=round(100 * sb[i], 1), partial=round(100 * cb[i], 1), n=nb[i]))
print("\n| model | median words | raw strict gain | = length part | + control-at-length part | strict AUC diff | partial AUC diff |")
for m in sys.argv[1:]:
    R = rows(m); pm, sm, cm, nm = stats(R, (pb, sb, cb)); s = summary(R, B); ci = boot(R, B)
    med = sorted(r["w"] for r in R)[len(R) // 2]
    f = lambda k: f"{100 * s[k]:+.1f} [{100 * ci[k][0]:+.1f}, {100 * ci[k][1]:+.1f}]"
    print(f"| {m} | {med} | {f('raw')} | {f('length')} | {f('control')} | {f('auc_s')} | {f('auc_c')} |")
    OUTD.append(dict(model=m, median_words=med, **{k: round(100 * v, 1) for k, v in s.items()}, **{f"{k}_lo": round(100 * ci[k][0], 1) for k in s}, **{f"{k}_hi": round(100 * ci[k][1], 1) for k in s}))
    for i, l in enumerate(LAB): OUTC.append(dict(model=m, bin=l, bin_i=i, share=round(pm[i], 3), strict=round(100 * sm[i], 1), partial=round(100 * cm[i], 1), n=nm[i]))
    print(f"|   curve | | share {['%.2f' % x for x in pm]} | strict {['%.0f' % (100 * x) for x in sm]} | partial {['%.0f' % (100 * x) for x in cm]} | | |")

D = os.path.join(O, "dash_lang")
json.dump(OUTD, open(f"{D}/length_decomp.json", "w"), indent=0); json.dump(OUTC, open(f"{D}/length_curves.json", "w"), indent=0)
