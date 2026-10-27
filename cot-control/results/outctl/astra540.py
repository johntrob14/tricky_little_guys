"""Paired comparison vs base on the CoT-Control items: the 108 Astra items (more/cot) and the 432 newer ones (big/cot), whatever a model has.
Per item: mean strict pass and mean continuous score (rl/astra_continuous.score: share of letters / word pairs / sentences / keywords in
compliance; ignore_question = strict) over that model's samples. Difference vs base per item, 95% bootstrap CI over items (5000 resamples, seed 0).
Run from cot-control/rl:  uv run python ../results/outctl/astra540.py MODEL [MODEL ...]"""
import ast, json, os, random, sys, collections
sys.path.insert(0, "."); sys.path.insert(0, "../grading")
import astra_continuous as AC
O = os.path.dirname(os.path.abspath(__file__))
def items(path):
    its = [json.loads(l) for l in open(path)] if path.endswith(".jsonl") else json.load(open(path))
    return {it["key"]: (ast.literal_eval(it["grader_args"]) if isinstance(it["grader_args"], str) else it["grader_args"]) for it in its}
GA = items(f"{O}/../tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl") | items(f"{O}/new_items.json")
def item_key(r): return r["key"].rsplit("|", 1)[0] if r["key"].startswith("astra|") else r["key"]
def per_item(m):
    s, c = collections.defaultdict(list), collections.defaultdict(list)
    for d in ("more", "big"):
        p = f"{O}/{d}/cot/{m}/graded.json"
        if not os.path.exists(p): continue
        for r in json.load(open(p)):
            k = r["key"]; ga = GA.get(k) or GA.get(item_key(r)) or {}
            st = bool(int(r["strict"])) if str(r["strict"]).isdigit() else bool(r["strict"])
            s[(d, k)].append(st); c[(d, k)].append(AC.score(dict(constraint=r["constraint"], reasoning=r.get("body") or "", grader_args=ga), st))
    return {k: sum(v) / len(v) for k, v in s.items()}, {k: sum(v) / len(v) for k, v in c.items()}, sum(len(v) for v in s.values())
def boot(x, y):
    ks = sorted(set(x) & set(y)); d = [x[k] - y[k] for k in ks]; rng = random.Random(0)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
    return f"{100*sum(d)/len(d):+.1f} [{100*bs[125]:+.1f}, {100*bs[4875]:+.1f}]", len(ks)
bs, bc, bn = per_item("base")
print("| model | items | samples | strict vs base (pts) | continuous vs base (pts) |")
for m in sys.argv[1:]:
    s, c, n = per_item(m); (ds, k), (dc, _) = boot(s, bs), boot(c, bc)
    print(f"| {m} | {k} | {n} | {ds} | {dc} |")
