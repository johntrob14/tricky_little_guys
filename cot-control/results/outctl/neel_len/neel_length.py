"""How much of Neel's 34-model CoT-Control result is CoT length? Reads ~/code/neel-astra/raw/cot_control_data/scratch_cot_control/graded/*.jsonl
(108 items per model, raw reasoning graded; compliance 1 = pass; reasoning_tokens per row). For each model: pass rate, median reasoning
tokens, pass rate by reasoning-length bin, and the pass rate reweighted to the pooled (all-model) length distribution over the bins the model covers ("length-matched"), with
Kitagawa split of (model - pooled) into length part + control-at-length part. Then Spearman of NCRI (models_ncri15_2.csv) and of CoT length with
raw and length-matched controllability.   python3 neel_length.py   -> neel_length.json"""
import json, glob, os, csv, collections, random
G = os.path.expanduser("~/code/neel-astra/raw/cot_control_data/scratch_cot_control/graded")
BINS = [(0, 250), (250, 600), (600, 1500), (1500, 4000), (4000, 10**9)]; LAB = ["<250", "250-600", "600-1500", "1500-4000", "4000+"]
rows = {}
for f in glob.glob(f"{G}/*.jsonl"):
    R = [json.loads(l) for l in open(f)]
    R = [r for r in R if r.get("compliance") in (0, 1) and r.get("reasoning_tokens") is not None]
    if R: rows[R[0]["model"]] = R
b = lambda t: next(i for i, (a, z) in enumerate(BINS) if a <= t < z)
pool = [r for R in rows.values() for r in R]
pw = [sum(1 for r in pool if b(r["reasoning_tokens"]) == i) / len(pool) for i in range(len(BINS))]
pc = [(sum(r["compliance"] for r in pool if b(r["reasoning_tokens"]) == i) / max(1, sum(1 for r in pool if b(r["reasoning_tokens"]) == i))) for i in range(len(BINS))]
ncri = {}
p = os.path.expanduser("~/code/neel-astra/raw/ncri_raw_data/models_ncri15_2.csv")
for r in csv.DictReader(open(p)):
    k = next((c for c in r if c.lower() in ("slug", "model", "model_id")), None)
    if k and r.get("ncri15_2"): ncri[r[k].replace("/", "_")] = float(r["ncri15_2"])
out = []
for m, R in sorted(rows.items()):
    n = [sum(1 for r in R if b(r["reasoning_tokens"]) == i) for i in range(len(BINS))]
    w = [x / len(R) for x in n]
    raw = sum(r["compliance"] for r in R) / len(R)
    c = [(sum(r["compliance"] for r in R if b(r["reasoning_tokens"]) == i) / n[i]) if n[i] else None for i in range(len(BINS))]
    cov = [i for i in range(len(BINS)) if c[i] is not None]  # length-matched: pooled weights over the bins this model covers, renormalized
    matched = sum(pw[i] * c[i] for i in cov) / sum(pw[i] for i in cov)
    c = [x if x is not None else raw for x in c]  # empty bins take the model's own rate in the decomposition
    length_part = sum((w[i] - pw[i]) * pc[i] for i in range(len(BINS))); control_part = sum(w[i] * (c[i] - pc[i]) for i in range(len(BINS)))
    med = sorted(r["reasoning_tokens"] for r in R)[len(R) // 2]
    out.append(dict(model=m, n=len(R), raw=round(100 * raw, 1), matched=round(100 * matched, 1), median_tokens=med, length_part=round(100 * length_part, 1), control_part=round(100 * control_part, 1),
                    share=[round(x, 3) for x in w], by_bin=[round(100 * x, 1) for x in c], bin_n=n, ncri=ncri.get(m.replace("/", "_"))))
def rank(v): s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); [r.__setitem__(i, k) for k, i in enumerate(s)]; return r
def spearman(a, b_):
    ra, rb = rank(a), rank(b_); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5; return num / den if den else float("nan")
pooled_raw = 100 * sum(r["compliance"] for r in pool) / len(pool)
print(f"pooled pass {pooled_raw:.2f}%  bins {LAB}  pooled share {[round(x, 2) for x in pw]}  pooled pass by bin {[round(100 * x, 1) for x in pc]}")
print("| model | median reasoning tokens | raw pass | length-matched pass | length part | control part | NCRI |")
for o in sorted(out, key=lambda o: -o["raw"]): print(f"| {o['model']} | {o['median_tokens']} | {o['raw']} | {o['matched']} | {o['length_part']:+} | {o['control_part']:+} | {o['ncri']} |")
X = [o for o in out if o["ncri"] is not None]
print(f"\nSpearman (n={len(X)}): NCRI vs raw {spearman([o['ncri'] for o in X], [o['raw'] for o in X]):+.2f}, NCRI vs length-matched {spearman([o['ncri'] for o in X], [o['matched'] for o in X]):+.2f}, "
      f"median length vs raw {spearman([o['median_tokens'] for o in out], [o['raw'] for o in out]):+.2f}, median length vs matched {spearman([o['median_tokens'] for o in out], [o['matched'] for o in out]):+.2f}")
print(f"share of between-model variance in raw pass explained by the length part: var(length)={sum((o['length_part']) ** 2 for o in out) / len(out):.2f} var(control)={sum((o['control_part']) ** 2 for o in out) / len(out):.2f}")
json.dump(dict(bins=LAB, pooled_share=pw, pooled_by_bin=[100 * x for x in pc], pooled_raw=pooled_raw, models=out), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "neel_length.json"), "w"), indent=1)
