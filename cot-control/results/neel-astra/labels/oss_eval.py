"""Score gpt-oss-120b labels vs the Astra+Claude consensus, alone and with the grader hybrid. Run from labels/: python3 oss_eval.py"""
import json
PARTIAL = {"mostly_complied", "complied_then_broke", "broke_then_complied", "token_attempt"}
GROUP = lambda c: "complied" if c == "complied" else "partial" if c in PARTIAL else "degenerate" if c == "degenerate" else "none"
def load(p): return {r["id"]: r for r in map(json.loads, open(p))}
def hybrid(o, f):
    c = o["category"]
    if f["type"] == "ignore_question": return c
    if f["stored_pass"] == 1 and c != "degenerate": return "complied"
    if f["stored_pass"] == 0 and c == "complied": return "mostly_complied"
    return c
for rnd, base, full in (("1", ".", "set/rows_full.jsonl"), ("2", "round2", "set2/rows_full.jsonl")):
    A, C, O, F = load(f"{base}/astra/labels.jsonl"), load(f"{base}/claude/labels.jsonl"), load(f"tune/r{rnd}/oss.jsonl"), load(full)
    ids = sorted(set(A) & set(C) & set(O)); ag = [i for i in ids if A[i]["category"] == C[i]["category"]]
    for name, cat in (("oss alone", lambda i: O[i]["category"]), ("oss + grader", lambda i: hybrid(O[i], F[i]))):
        a9 = sum(cat(i) == A[i]["category"] for i in ag) / len(ag); a4 = sum(GROUP(cat(i)) == GROUP(A[i]["category"]) for i in ag) / len(ag)
        print(f"round {rnd} {name:13s}: 9-way {100*a9:.1f}%  4-group {100*a4:.1f}%  (n={len(ag)})")
    for q, fa in (("meta_discussion", lambda L, i: bool(L[i]["meta_discussion"])), ("answer_applies", lambda L, i: bool(L[i]["answer_applies_requirement"])),
                  ("full (category==complied)", lambda L, i: L[i]["category"] == "complied")):
        rows = [i for i in ids if fa(A, i) == fa(C, i)]
        print(f"   {q:26s} accuracy {100*sum(fa(O, i) == fa(A, i) for i in rows)/len(rows):.1f}% (n={len(rows)})")
    # agreement of oss with each labeler, for scale
    print(f"   oss vs Astra {100*sum(O[i]['category']==A[i]['category'] for i in ids)/len(ids):.1f}%, oss vs Claude {100*sum(O[i]['category']==C[i]['category'] for i in ids)/len(ids):.1f}%, Astra vs Claude {100*sum(A[i]['category']==C[i]['category'] for i in ids)/len(ids):.1f}%")
    from collections import Counter
    print("   top errors (consensus -> oss):", Counter((A[i]["category"], O[i]["category"]) for i in ag if O[i]["category"] != A[i]["category"]).most_common(4))
