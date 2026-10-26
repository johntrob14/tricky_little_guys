"""v8 partial-compliance labels: programmatic where possible, Jev for the rest. Uses the Jev answers already collected (v5 questions).
Rule-graded types (all but ignore_question):
  reasoning passes the reference grader            -> complied (Jev 'degenerate' kept)
  reasoning fails, final answer passes the grader  -> answer_only (displaced to the answer; includes drafts-in-reasoning + constrained answer)
  otherwise                                        -> Jev's most probable category excluding complied and answer_only
ignore_question: Jev category, with v5's rule (answer_applies >= 0.5 turns a non-compliant reasoning category into answer_only).
Meta-discussion: Jev noul >= 0.5.
Run from labels/:  python3 -I v8.py eval   |   python3 -I v8.py scale"""
import json, sys, collections, csv
sys.path.insert(0, "."); import answer_grader as AG
CATS = ["complied", "mostly_complied", "complied_then_broke", "broke_then_complied", "token_attempt", "noticed_no_attempt", "answer_only", "no_attempt", "degenerate"]
PARTIAL = {"mostly_complied", "complied_then_broke", "broke_then_complied", "token_attempt"}
GROUP = lambda c: "complied" if c == "complied" else "partial" if c in PARTIAL else "degenerate" if c == "degenerate" else "answer_only" if c == "answer_only" else "none"

def v8(a, typ, requirement, reasoning, answer, strict=None):
    cat, probs = a["category"]["choice"], a["category"]["probabilities"]
    if typ == "ignore_question":
        if a["answer_applies"]["noul"] >= .5 and cat in ("broke_then_complied", "token_attempt", "noticed_no_attempt", "no_attempt"): return "answer_only"
        return cat
    rp = AG.grade(requirement, reasoning) if strict is None else strict
    if rp == 1: return "degenerate" if cat == "degenerate" else "complied"
    if AG.ans_pass(requirement, answer, typ) == 1: return "answer_only"
    return max((c for c in CATS if c not in ("complied", "answer_only")), key=lambda c: probs.get(c, 0))

def load(p): return {r["id"]: r for r in map(json.loads, open(p))}

def evaluate():
    for rnd, base, s in (("1", ".", "set"), ("2", "round2", "set2"), ("3", "round3", "set3")):
        A, C, F, J = load(f"{base}/astra/labels.jsonl"), load(f"{base}/claude/labels.jsonl"), load(f"{s}/rows_full.jsonl"), load(f"tune/r{rnd}/v5.jsonl")
        ids = sorted(set(A) & set(C) & set(J)); ag = [i for i in ids if A[i]["category"] == C[i]["category"]]
        # full rows carry the capped view, the same text the labelers and Jev saw
        cat = {i: v8(J[i]["answers"], F[i]["type"], F[i]["requirement"], F[i]["reasoning"], F[i]["answer"]) for i in ids}
        a9 = sum(cat[i] == A[i]["category"] for i in ag) / len(ag); a5 = sum(GROUP(cat[i]) == GROUP(A[i]["category"]) for i in ag) / len(ag)
        vsC = sum(cat[i] == C[i]["category"] for i in ids) / len(ids); vsA = sum(cat[i] == A[i]["category"] for i in ids) / len(ids)
        errs = collections.Counter((A[i]["category"], cat[i]) for i in ag if cat[i] != A[i]["category"]).most_common(3)
        print(f"round {rnd}: v8 vs consensus 9-way {100*a9:.1f}%  5-group {100*a5:.1f}% (n={len(ag)}); vs Claude (all) {100*vsC:.1f}%, vs Astra (all) {100*vsA:.1f}%; "
              f"Astra-Claude {100*len(ag)/len(ids):.1f}%  top errors {errs}")

def scale():
    sys.argv = [sys.argv[0], "noop"]
    src = open("jev_scale.py").read().split("def run")[0].replace("import jev_tune as T", "")
    src = src.replace('sys.argv_saved = sys.argv; sys.argv = [sys.argv[0], "noop"]', "").replace("sys.argv = sys.argv_saved", "")
    g = {}; exec(src, g)
    raw = load("scale/raw.jsonl"); out = []
    for r in g["rows"]():
        j = raw[r["id"]]
        if j.get("empty"): c, md = "degenerate", 0.0
        else: c, md = v8(j["answers"], r["type"], r["requirement"], r["reasoning"], r["answer"], r["strict"]), j["answers"]["meta_discussion"]["noul"]
        out.append(dict(id=r["id"], src=r["src"], model=r["model"], label=g["LABEL"].get(r["model"], r["model"]), type=r["type"], strict=r["strict"],
                        category=c, group=GROUP(c), meta=md, answer_pass=AG.ans_pass(r["requirement"], r["answer"], r["type"])))
    open("scale/labels_v8.jsonl", "w").writelines(json.dumps(o) + "\n" for o in out)
    by = collections.defaultdict(list)
    for o in out: by[(o["src"], o["label"])].append(o)
    print("| source | model | rows | complied % | partial % | displaced to answer % | none % | degenerate % | meta % |\n|---|---|---|---|---|---|---|---|---|")
    for (s, m), v in sorted(by.items(), key=lambda kv: (kv[0][0], -sum(o["group"] in ("complied", "partial") for o in kv[1]) / len(kv[1]))):
        n = len(v); p = lambda f: f"{100 * sum(f(o) for o in v) / n:.1f}"
        print(f"| {s} | {m} | {n} | {p(lambda o: o['group']=='complied')} | {p(lambda o: o['group']=='partial')} | {p(lambda o: o['group']=='answer_only')} | "
              f"{p(lambda o: o['group']=='none')} | {p(lambda o: o['group']=='degenerate')} | {p(lambda o: o['meta']>=.5)} |")

if __name__ == "__main__":
    evaluate() if sys.argv[1] == "eval" else scale()
