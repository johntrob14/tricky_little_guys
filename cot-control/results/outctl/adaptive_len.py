"""Adaptive shortening: per CoT-Control item, the model's CoT words under the constraint (mean over its samples, cot side) divided by the same
model's CoT words on the same question with no constraint (plain_len.jsonl). Reports the model's natural (unconstrained) median length, its
constrained median, and the geometric-mean ratio with a 95% bootstrap CI over items (items with a 0-word plain CoT are skipped).
python3 adaptive_len.py MODEL [...]"""
import json, math, random, sys, collections, os
O = os.path.dirname(os.path.abspath(__file__))
P = collections.defaultdict(dict)
for r in map(json.loads, open(f"{O}/plain_len.jsonl")):
    for k in r["keys"]: P[r["model"]][k] = r["words"]
def cons(m):
    d = collections.defaultdict(list)
    for dd in ("more", "big"):
        p = f"{O}/{dd}/cot/{m}/graded.json"
        if os.path.exists(p):
            for r in json.load(open(p)): d[r["key"].rsplit("|", 1)[0] if r["key"].startswith("astra|") else r["key"]].append(int(r["words"] or 0))
    return {k: sum(v) / len(v) for k, v in d.items()}
print("| model | unconstrained median words | constrained median words | constrained / unconstrained (geo-mean) [95% CI] |")
out = []
for m in sys.argv[1:]:
    C = cons(m); keys = [k for k in C if P[m].get(k, 0) > 0]
    lr = [math.log(max(C[k], 1) / P[m][k]) for k in keys]; rng = random.Random(0)
    bs = sorted(sum(rng.choice(lr) for _ in lr) / len(lr) for _ in range(2000))
    u = sorted(P[m][k] for k in keys); c = sorted(C[k] for k in keys)
    row = dict(model=m, items=len(keys), plain_median=u[len(u) // 2], constrained_median=round(c[len(c) // 2]), ratio=round(math.exp(sum(lr) / len(lr)), 3), lo=round(math.exp(bs[50]), 3), hi=round(math.exp(bs[1950]), 3))
    out.append(row); print(f"| {m} | {row['plain_median']} | {row['constrained_median']} | {row['ratio']:.2f} [{row['lo']:.2f}, {row['hi']:.2f}] |")
json.dump(out, open(f"{O}/adaptive_len.json", "w"), indent=1)
