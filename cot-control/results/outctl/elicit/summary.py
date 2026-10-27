"""Table of elicit runs: per model x variant x split, strict pass on the 8 non-ignore types (constraint-balanced), partial compliance (same),
ignore_question pass, accuracy (ANSWER letter in the response), median reasoning words, cap-outs.   python3 elicit/summary.py [split]"""
import glob, json, os, re, sys, collections
O = os.path.dirname(os.path.abspath(__file__)); SP = sys.argv[1] if len(sys.argv) > 1 else "dev"
def bal(G, f):
    by = collections.defaultdict(list)
    for g in G: by[g["constraint"]].append(f(g))
    return 100 * sum(sum(v) / len(v) for v in by.values()) / len(by) if by else float("nan")
rows = []
for d in sorted(glob.glob(f"{O}/runs/*/*_{SP}")):
    if not os.path.exists(f"{d}/graded.json"): continue
    G = json.load(open(f"{d}/graded.json")); m, v = d.split("/")[-2], os.path.basename(d)[: -len(SP) - 1]
    N = [g for g in G if g["constraint"] != "ignore_question"]; I = [g for g in G if g["constraint"] == "ignore_question"]
    acc = 100 * sum((lambda x: bool(x) and x[-1] == g["gold"])(re.findall(r"ANSWER:\s*\**\s*([A-J])\b", g.get("resp") or "")) for g in G) / len(G)
    w = sorted(g["words"] for g in G)
    rows.append(dict(model=m, variant=v, n=len(G), strict=round(bal(N, lambda g: g["strict"]), 1), partial=round(100 * bal(N, lambda g: g["partial"]) / 100, 1),
                     ignore=round(100 * sum(g["strict"] for g in I) / len(I), 0) if I else None, acc=round(acc, 0), words=w[len(w) // 2], capped=sum(g["stop"] == "length" for g in G),
                     per={c[:8]: round(100 * sum(g["strict"] for g in N if g["constraint"] == c) / max(1, sum(g["constraint"] == c for g in N))) for c in sorted({g["constraint"] for g in N})}))
print(f"split {SP}\n| model | variant | n | strict (8 types) | partial | ignore_q | acc | median words | capped |")
for r in rows: print(f"| {r['model']} | {r['variant']} | {r['n']} | {r['strict']} | {r['partial']} | {r['ignore']} | {r['acc']} | {r['words']} | {r['capped']} |")
if "-v" in sys.argv:
    for r in rows: print(r["model"], r["variant"], r["per"])
json.dump(rows, open(f"{O}/summary_{SP}.json", "w"), indent=1)
