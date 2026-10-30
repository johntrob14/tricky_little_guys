"""Score round 2 against PREREG.md. Run from labels/round2/:  python3 score_prereg.py"""
import json, collections

def load(p): return {r["id"]: r for r in map(json.loads, open(p))}
A, C, J, F = load("astra/labels.jsonl"), load("claude/labels.jsonl"), load("jev/labels.jsonl"), load("../set2/rows_full.jsonl")
ids = sorted(set(A) & set(C) & set(J))
PARTIAL = {"mostly_complied", "complied_then_broke", "broke_then_complied", "token_attempt"}
GROUP = lambda c: "complied" if c == "complied" else "partial" if c in PARTIAL else "degenerate" if c == "degenerate" else "none"
ans = lambda i: J[i]["answers"]
def auc(pos, neg):
    return sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")
print(f"rows with all three labels: {len(ids)}")
ag = [i for i in ids if A[i]["category"] == C[i]["category"]]
print(f"Astra-Claude category agreement: {len(ag)}/{len(ids)} = {100*len(ag)/len(ids):.1f}%  "
      f"(4 groups {100*sum(GROUP(A[i]['category'])==GROUP(C[i]['category']) for i in ids)/len(ids):.1f}%)")
acc = [i for i in ag if ans(i)["category"]["confidence"] >= .70]
print(f"\nRule 1 (accept category at confidence >= 0.70): coverage {len(acc)}/{len(ag)} = {100*len(acc)/len(ag):.1f}% of consensus rows; "
      f"9-way {100*sum(ans(i)['category']['choice']==A[i]['category'] for i in acc)/len(acc):.1f}%, "
      f"4-group {100*sum(GROUP(ans(i)['category']['choice'])==GROUP(A[i]['category']) for i in acc)/len(acc):.1f}%   [round 1: 81/161, 85.2% 9-way]")
print(f"  all consensus rows, no threshold: 9-way {100*sum(ans(i)['category']['choice']==A[i]['category'] for i in ag)/len(ag):.1f}%   [round 1: 52.8%]")
for n, (q, flag) in enumerate((("full_compliance", lambda L, i: L[i]["category"] == "complied"),
                               ("meta_discussion", lambda L, i: bool(L[i]["meta_discussion"])),
                               ("answer_applies", lambda L, i: bool(L[i]["answer_applies_requirement"]))), 2):
    rows = [i for i in ids if flag(A, i) == flag(C, i)]
    pos = [ans(i)[q]["noul"] for i in rows if flag(A, i)]; neg = [ans(i)[q]["noul"] for i in rows if not flag(A, i)]
    print(f"Rule {n} ({q} >= 0.5): AUC {auc(pos, neg):.3f}, accuracy {100*sum((ans(i)[q]['noul']>=.5)==flag(A,i) for i in rows)/len(rows):.1f}% "
          f"(n={len(rows)}, positives {len(pos)}; consensus on flag {len(rows)}/{len(ids)})")
rows = [i for i in ids if (GROUP(A[i]["category"]) == "partial") == (GROUP(C[i]["category"]) == "partial")]
pp = {i: sum(ans(i)["category"]["probabilities"].get(c, 0) for c in PARTIAL) for i in ids}
print(f"Rule 5 (partial, sum of partial probs): AUC {auc([pp[i] for i in rows if GROUP(A[i]['category'])=='partial'], [pp[i] for i in rows if GROUP(A[i]['category'])!='partial']):.3f} "
      f"(n={len(rows)})   [round 1: 0.816]")
print("\nfor reference, round-1 Jev on these flags: full 0.980 / 92.2%, meta 0.995 / 97.0%, answer 0.936 / 88.4%")
