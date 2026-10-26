"""Score Jev's labels against the Astra + Claude labels. Reference = rows where Astra and Claude agree.
Run from labels/:  python3 jev_eval.py"""
import json, collections, statistics as st

def load(p): return {r["id"]: r for r in map(json.loads, open(p))}
A, C, J, F = load("astra/labels.jsonl"), load("claude/labels.jsonl"), load("jev/labels.jsonl"), load("set/rows_full.jsonl")
ids = sorted(J)
PARTIAL = {"mostly_complied", "complied_then_broke", "broke_then_complied", "token_attempt"}
GROUP = lambda c: "complied" if c == "complied" else "partial" if c in PARTIAL else "degenerate" if c == "degenerate" else "none"
jc = {i: J[i]["answers"]["category"]["choice"] for i in ids}
jconf = {i: J[i]["answers"]["category"]["confidence"] for i in ids}
nou = lambda i, q: J[i]["answers"][q]["noul"]

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

agree = [i for i in ids if A[i]["category"] == C[i]["category"]]
print(f"Astra-Claude agree on category in {len(agree)}/200 rows (the reference set)\n")
print("category accuracy:")
for name, ref, rows in (("vs consensus (agreed rows)", lambda i: A[i]["category"], agree), ("vs Astra (all)", lambda i: A[i]["category"], ids),
                        ("vs Claude (all)", lambda i: C[i]["category"], ids)):
    exact = sum(jc[i] == ref(i) for i in rows) / len(rows); grp = sum(GROUP(jc[i]) == GROUP(ref(i)) for i in rows) / len(rows)
    print(f"  {name:28s} 9-way {100*exact:5.1f}%   4 groups {100*grp:5.1f}%   (n={len(rows)})")
print("  for scale: Astra vs Claude 9-way 80.5%, 4 groups 87.0%\n")

print("binary questions vs consensus (rows where Astra and Claude agree on that flag):")
for q, flag in (("full_compliance", lambda L, i: L[i]["category"] == "complied"),
                ("meta_discussion", lambda L, i: bool(L[i]["meta_discussion"])),
                ("answer_applies", lambda L, i: bool(L[i]["answer_applies_requirement"]))):
    rows = [i for i in ids if flag(A, i) == flag(C, i)]
    pos = [nou(i, q) for i in rows if flag(A, i)]; neg = [nou(i, q) for i in rows if not flag(A, i)]
    acc = sum((nou(i, q) >= .5) == flag(A, i) for i in rows) / len(rows)
    print(f"  {q:16s} AUC {auc(pos, neg):.3f}  accuracy at 0.5 {100*acc:5.1f}%  (n={len(rows)}, positives {len(pos)})")
# partial compliance as a group, from the category probabilities
rows = [i for i in ids if (GROUP(A[i]["category"]) == "partial") == (GROUP(C[i]["category"]) == "partial")]
pp = {i: sum(J[i]["answers"]["category"]["probabilities"].get(c, 0) for c in PARTIAL) for i in ids}
pos = [pp[i] for i in rows if GROUP(A[i]["category"]) == "partial"]; neg = [pp[i] for i in rows if GROUP(A[i]["category"]) != "partial"]
print(f"  partial (sum of 4 partial-category probs) AUC {auc(pos, neg):.3f}  (n={len(rows)}, positives {len(pos)})")
fc = [nou(i, "full_compliance") for i in ids]
print(f"  full_compliance vs official grader: AUC {auc([nou(i,'full_compliance') for i in ids if F[i]['stored_pass']], [nou(i,'full_compliance') for i in ids if not F[i]['stored_pass']]):.3f}\n")

print("calibration of Jev's category confidence (vs consensus, agreed rows):")
for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
    b = [i for i in agree if lo <= jconf[i] < hi]
    if b: print(f"  confidence {lo:.1f}-{min(hi,1):.1f}: n={len(b):3d}  9-way accuracy {100*sum(jc[i]==A[i]['category'] for i in b)/len(b):5.1f}%  4-group {100*sum(GROUP(jc[i])==GROUP(A[i]['category']) for i in b)/len(b):5.1f}%")
dis = [i for i in ids if A[i]["category"] != C[i]["category"]]
print(f"\nJev confidence where Astra and Claude agree: median {st.median(jconf[i] for i in agree):.2f}; where they disagree: median {st.median(jconf[i] for i in dis):.2f}")
print(f"AUC of (1 - confidence) for predicting an Astra-Claude disagreement: {auc([1-jconf[i] for i in dis], [1-jconf[i] for i in agree]):.3f}")
print("\nJev 9-way confusion on agreed rows (row = consensus, col = Jev):")
cats = ["complied", "mostly_complied", "complied_then_broke", "broke_then_complied", "token_attempt", "noticed_no_attempt", "answer_only", "no_attempt", "degenerate"]
print("".ljust(20) + "".join(c[:9].rjust(10) for c in cats))
for a in cats:
    print(a[:19].ljust(20) + "".join(str(sum(1 for i in agree if A[i]["category"] == a and jc[i] == c) or ".").rjust(10) for c in cats))
