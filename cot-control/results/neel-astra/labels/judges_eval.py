"""Compare cheap OpenRouter judges, Jev and gpt-oss-120b on rounds 1-3 against the Astra+Claude consensus.
Alone: the judge's own category. Hybrid (v8 frame): grader pass -> complied, reasoning fail + answer passes grader -> answer_only,
otherwise the judge's category (a judge 'complied' there becomes mostly_complied; a judge 'answer_only' there is kept).
Jev's hybrid is v8 itself (it picks the next most probable category instead). Run from labels/: python3 -I judges_eval.py"""
import json, sys, glob, os, collections
sys.path.insert(0, "."); import answer_grader as AG
from v8 import v8 as jev_v8, GROUP
def load(p): return {r["id"]: r for r in map(json.loads, open(p))}
RND = (("1", ".", "set"), ("2", "round2", "set2"), ("3", "round3", "set3"))
REF = {r: (load(f"{b}/astra/labels.jsonl"), load(f"{b}/claude/labels.jsonl"), load(f"{s}/rows_full.jsonl")) for r, b, s in RND}
def hybrid(c, f):
    if f["type"] == "ignore_question": return c
    if AG.grade(f["requirement"], f["reasoning"]) == 1: return "degenerate" if c == "degenerate" else "complied"
    if AG.ans_pass(f["requirement"], f["answer"], f["type"]) == 1: return "answer_only"
    return "mostly_complied" if c == "complied" else c
names = sorted({os.path.basename(p)[:-6] for p in glob.glob("tune/r1/*.jsonl")} - {"v0", "v1", "v2", "v3", "v4", "v6", "v7"})
rows = []
for n in names:
    stats = collections.defaultdict(list); cost = 0; fails = 0; tot = 0
    for r, _, _ in RND:
        p = f"tune/r{r}/{n}.jsonl"
        if not os.path.exists(p): continue
        A, C, F = REF[r]; J = load(p)
        for i in A:
            tot += 1
            j = J.get(i)
            if not j or "error" in j: fails += 1; continue
            cost += j.get("_cost", 0) or j.get("cost", 0) or 0
            cons = A[i]["category"] == C[i]["category"]
            if n == "v5":
                alone = j["answers"]["category"]["choice"]; hyb = jev_v8(j["answers"], F[i]["type"], F[i]["requirement"], F[i]["reasoning"], F[i]["answer"])
                meta = j["answers"]["meta_discussion"]["noul"] >= .5; ans = j["answers"]["answer_applies"]["noul"] >= .5
            else:
                alone = j["category"]; hyb = hybrid(alone, F[i]); meta = bool(j["meta_discussion"]); ans = bool(j["answer_applies_requirement"])
            if cons:
                stats[f"alone{r}"].append(alone == A[i]["category"]); stats[f"hyb{r}"].append(hyb == A[i]["category"])
                stats["hyb5"].append(GROUP(hyb) == GROUP(A[i]["category"]))
            if bool(A[i]["meta_discussion"]) == bool(C[i]["meta_discussion"]): stats["meta"].append(meta == bool(A[i]["meta_discussion"]))
            if bool(A[i]["answer_applies_requirement"]) == bool(C[i]["answer_applies_requirement"]): stats["ans"].append(ans == bool(A[i]["answer_applies_requirement"]))
    pc = lambda k: f"{100 * sum(stats[k]) / len(stats[k]):.1f}" if stats[k] else "-"
    rows.append((n.replace("_", "/", 1) if n != "v5" else "Jev 1.13 (v5 questions)", pc("alone1"), pc("alone2"), pc("alone3"), pc("hyb1"), pc("hyb2"), pc("hyb3"), pc("hyb5"), pc("meta"), pc("ans"), f"{fails}/{tot}", f"${cost:.3f}"))
print("| judge | alone R1 | alone R2 | alone R3 | hybrid R1 | hybrid R2 | hybrid R3 | hybrid 5-group (all) | meta acc | answer-applies acc | failed rows | cost (500 rows) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: -float(r[7]) if r[7] != "-" else 0): print("| " + " | ".join(r) + " |")
