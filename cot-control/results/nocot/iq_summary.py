import json, csv, statistics as st, collections, random
csv.field_size_limit(10**9)
rows = json.load(open("nocot/st1_ignore2_iq_rows.json"))
nc = collections.defaultdict(list)
for r in map(json.loads, open("nocot/rows.jsonl")): nc[(r["src"], r["arm"], r["dataset"], r["item_idx"])].append(r["correct"])
ncq = lambda src, arm, d, i: st.mean(nc[(src, arm, d, i)])
st1_noreq = {(r["dataset"], r["item_idx"]): r["correct"] for r in map(json.loads, open("transfer/gptoss-st1-ignore2-iq/gens.jsonl")) if r["type"] == "baseline"}
bg = [json.loads(l) for l in open("transfer/gptoss-base/hf/evals/transfer/gptoss-base/gens.jsonl")]
base_noreq = {(r["dataset"], r["item_idx"]): r["correct"] for r in bg if r["type"] == "baseline"}
base_iq = {(r["dataset"], r["item_idx"]): r["correct"] for r in bg if r["type"] == "ignore_question"}
bj = {(r["dataset"], r["item_idx"]): r["score"] == 1 for r in map(json.loads, open("transfer/gptoss-base/hf/evals/transfer/gptoss-base/judgments.jsonl")) if r.get("kind") == "compliance" and r.get("constraint") == "ignore_question"}
def boot(diffs, n=4000):
    rng = random.Random(0); m = [st.mean(rng.choice(diffs) for _ in diffs) for _ in range(n)]; m.sort(); return m[int(.025*n)], m[int(.975*n)]
def report(name, sub, src):
    if not sub: print(name, "n=0"); return
    acc = st.mean(x["correct"] for x in sub)
    noc = st.mean(ncq(src, "ignore", x["dataset"], x["item_idx"]) for x in sub)
    nocq = st.mean(ncq(src, "question", x["dataset"], x["item_idx"]) for x in sub)
    d = [x["correct"] - ncq(src, "ignore", x["dataset"], x["item_idx"]) for x in sub]
    lo, hi = boot(d)
    extra = ""
    if src == "transfer":
        k = [(x["dataset"], x["item_idx"]) for x in sub]
        extra = f" | st1-ignore2 no-req {st.mean(st1_noreq[i] for i in k):.3f} | base CoT no-req {st.mean(base_noreq[i] for i in k):.3f}"
    print(f"{name:55s} n={len(sub):3d} st1-ignore2 acc {acc:.3f} | no-CoT (ignore prompt) {noc:.3f} (question prompt {nocq:.3f}) | diff {acc-noc:+.3f} [{lo:+.3f},{hi:+.3f}]{extra}")
for src, levels in (("transfer", ["medium"]), ("astra", ["low", "medium", "high"])):
    for lv in levels + (["all"] if src == "astra" else []):
        g = [x for x in rows if x["src"] == src and (lv == "all" or x["level"] == lv)]
        print(f"\n== {src} {lv}: compliance strict {sum(x['strict'] for x in g)}/{len(g)}, attempt-judge YES low {sum(x['attempt_low'] is True for x in g)} high {sum(x['attempt_high'] is True for x in g)}, median CoT words {st.median(len(x['reasoning'].split()) for x in g)}")
        report("all rows", g, src)
        report("strict & attempt NO (low effort)", [x for x in g if x["strict"] and x["attempt_low"] is False], src)
        report("strict & attempt NO (high effort)", [x for x in g if x["strict"] and x["attempt_high"] is False], src)
        report("strict & attempt NO at both efforts", [x for x in g if x["strict"] and x["attempt_low"] is False and x["attempt_high"] is False], src)
        report("strict & attempt YES (either)", [x for x in g if x["strict"] and (x["attempt_low"] or x["attempt_high"])], src)
print(f"\nbase gpt-oss normal CoT, transfer ignore_question: compliance {sum(bj.values())}/{len(bj)}, accuracy {sum(base_iq.values())}/108; no-req accuracy {sum(base_noreq.values())}/108")
for lv, run in (("low","astra-base-low-t1"),("medium","gptoss120b-t1"),("high","astra-base-high-t1")):
    rec=[r for r in csv.DictReader(open(f"tinker-runs/{run}/analysis/records.csv")) if r["constraint"]=="ignore_question" and r["protocol"]=="astra"]
    print(f"base astra {lv}: compliance {sum(r['strict']=='1' for r in rec)}/12 accuracy {sum(r['correct']=='True' for r in rec)}/12")
