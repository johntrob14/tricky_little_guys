"""Meta-discussion rate (Jev 1.13, v5 questions, noul >= 0.5) on CoT-arm rows for base, lang1-s40, manytypes and grpo3: overall and
among rows passing the strict grader. 300 rows per model (seed 0), all 9 types. Output: scale/meta_lang.jsonl"""
import json, random, sys, os, concurrent.futures as cf, collections
sys.argv = [sys.argv[0], "noop"]; import jev_tune as T
O = "/home/john/code/tlg-arena-rl/cot-control/results/outctl"
def req(p): return "Requirement:" + p.split("Requirement:", 1)[1]
rows = []
for m in ("base", "lang1-s40", "mt1b-s150", "grpo3"):
    xs = [x for d in ("more", "big") for x in json.load(open(f"{O}/{d}/cot/{m}/graded.json"))]
    random.Random(0).shuffle(xs)
    for x in xs[:300]:
        rows.append(dict(id=f"{m}|{x['key']}|{x['sample_idx']}", model=m, type=x["constraint"], strict=int(x["strict"]), requirement=req(x["user"]), reasoning=x["out"] or "", answer=x.get("resp") or ""))
with cf.ThreadPoolExecutor(12) as ex: res = list(ex.map(lambda r: T.call(r, "v5"), rows))
with open("scale/meta_lang.jsonl", "w") as f:
    for r, j in zip(rows, res): f.write(json.dumps(dict(r, jev=j.get("answers"), cost=j.get("cost"), error=j.get("error"))) + "\n")
print("cost $", round(sum(j.get("cost") or 0 for j in res), 4), "errors", sum("error" in j for j in res))
by = collections.defaultdict(list)
for r, j in zip(rows, res):
    if "answers" in j: by[r["model"]].append((r["strict"], j["answers"]["meta_discussion"]["noul"] >= .5))
for m, v in by.items():
    p = [mt for s, mt in v if s]; print(f"{m:10s} meta-discussion {100*sum(mt for _, mt in v)/len(v):5.1f}% of {len(v)} rows; among strict passes {100*sum(p)/max(1,len(p)):5.1f}% of {len(p)}")
