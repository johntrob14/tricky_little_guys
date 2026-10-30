"""Compare GPT-6 Astra and Claude partial-compliance labels on the 200-row set, and both against the official strict grade.
Run from labels/:  python3 compare.py  -> prints tables, writes disagreements.jsonl"""
import json, collections, statistics as st

def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}

A, C = load("astra/labels.jsonl"), load("claude/labels.jsonl")
F = load("set/rows_full.jsonl")
ids = sorted(set(A) & set(C))
print(f"rows labeled by both: {len(ids)} (astra {len(A)}, claude {len(C)})")

def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")

GROUP = {"complied": "complied", "mostly_complied": "partial", "complied_then_broke": "partial", "broke_then_complied": "partial",
         "token_attempt": "partial", "noticed_no_attempt": "none", "answer_only": "none", "no_attempt": "none", "degenerate": "degenerate"}
for field, f in (("category", lambda r: r["category"]), ("category, 4 groups", lambda r: GROUP.get(r["category"], "?")),
                 ("first_break", lambda r: r["first_break"]), ("meta_discussion", lambda r: bool(r["meta_discussion"])),
                 ("answer_applies_requirement", lambda r: bool(r["answer_applies_requirement"]))):
    po, k = kappa([(f(A[i]), f(C[i])) for i in ids])
    print(f"{field:28s} agreement {100*po:5.1f}%  kappa {k:.2f}")
sa, sc = [float(A[i]["score"]) for i in ids], [float(C[i]["score"]) for i in ids]
d = [abs(a - c) for a, c in zip(sa, sc)]
ma, mc = st.mean(sa), st.mean(sc)
r = sum((a - ma) * (c - mc) for a, c in zip(sa, sc)) / (sum((a - ma) ** 2 for a in sa) * sum((c - mc) ** 2 for c in sc)) ** .5
print(f"score: mean astra {ma:.2f} claude {mc:.2f}; mean |diff| {st.mean(d):.2f}; |diff|>=0.3 in {sum(x >= .3 for x in d)} rows; Pearson r {r:.2f}")

print("\ncategory confusion (rows = Astra, cols = Claude)")
cats = sorted({A[i]["category"] for i in ids} | {C[i]["category"] for i in ids})
ab = {c[:12] for c in cats}
print("astra \\ claude".ljust(22) + "".join(c[:10].rjust(11) for c in cats))
for a in cats:
    print(a.ljust(22) + "".join(str(sum(1 for i in ids if A[i]["category"] == a and C[i]["category"] == c) or ".").rjust(11) for c in cats))

print("\nagainst the official strict grade (stored_pass)")
for name, L in (("astra", A), ("claude", C)):
    t = collections.Counter((F[i]["stored_pass"], L[i]["category"] == "complied") for i in ids)
    print(f"{name}: grader pass & labeled complied {t[(1, True)]}, grader pass & not complied {t[(1, False)]}, "
          f"grader fail & labeled complied {t[(0, True)]}, grader fail & not complied {t[(0, False)]}")

print("\ncategory agreement by type and by source")
for key in ("type", "src"):
    g = collections.defaultdict(list)
    for i in ids: g[F[i][key]].append(A[i]["category"] == C[i]["category"])
    print(key, {k: f"{sum(v)}/{len(v)}" for k, v in sorted(g.items())})

with open("disagreements.jsonl", "w") as f:
    for i in ids:
        if A[i]["category"] != C[i]["category"] or abs(float(A[i]["score"]) - float(C[i]["score"])) >= .3:
            f.write(json.dumps(dict(id=i, type=F[i]["type"], src=F[i]["src"], model=F[i]["model"], stored_pass=F[i]["stored_pass"],
                                    astra={k: A[i][k] for k in ("category", "score", "first_break", "confidence", "note")},
                                    claude={k: C[i][k] for k in ("category", "score", "first_break", "confidence", "note")})) + "\n")
print("\nwrote disagreements.jsonl")
