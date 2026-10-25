import asyncio, csv, json, re, sys, statistics as st, collections
sys.path.insert(0, ".")
from grpo import JUDGE_PROMPT
from orjudge import OpenRouterJudge
csv.field_size_limit(10**9)
R = "../results"
rows = []
for r in map(json.loads, open(f"{R}/transfer/gptoss-st1-ignore2-iq/gens.jsonl")):
    if r["type"] == "ignore_question":
        rows.append(dict(src="transfer", level="medium", dataset=r["dataset"], item_idx=r["item_idx"], reasoning=r["reasoning"] or "", correct=r["correct"], q=r["qblock"]))
strictT = {}
for r in map(json.loads, open(f"{R}/transfer/gptoss-st1-ignore2-iq/judgments.jsonl")):
    if r.get("kind") == "compliance": strictT[(r["dataset"], r["item_idx"])] = r["score"] == 1
for x in rows: x["strict"] = strictT[(x["dataset"], x["item_idx"])]
for lv in ("low", "medium", "high"):
    run = f"{R}/tinker-runs/astra-st1-ignore2-{lv}-t1"
    rec = {(r["dataset"], r["item_idx"]): r for r in csv.DictReader(open(f"{run}/analysis/records.csv")) if r["constraint"] == "ignore_question"}
    for l in open(f"{run}/generations/gpt-oss-120b/astra.jsonl"):
        x = json.loads(l)
        if x["constraint"] != "ignore_question": continue
        k = (x["dataset"], str(x["item_idx"]))
        rows.append(dict(src="astra", level=lv, dataset=x["dataset"], item_idx=x["item_idx"], reasoning=x["reasoning"] or "", correct=rec[k]["correct"] == "True",
                         strict=rec[k]["strict"] == "1", q=x["user_prompt"].split("\n\n\nYour response should")[0]))
async def go():
    out = {}
    for eff in ("low", "high"):
        j = OpenRouterJudge("openai/gpt-oss-120b", effort=eff, concurrency=64, provider="DeepInfra")
        async def one(x):
            f, _, _ = await j.ask(JUDGE_PROMPT.format(question=x["q"].split("\n\n\nYour response should")[0], reasoning=x["reasoning"]))
            m = re.match(r"\W*(?:ANSWER:)?\W*(YES|NO)\b", f.upper()); return (m.group(1) == "YES") if m else None
        out[eff] = await asyncio.gather(*[one(x) for x in rows])
        print(eff, "cost", round(j.usd, 3), "unparsed", sum(v is None for v in out[eff]), file=sys.stderr)
    return out
v = asyncio.run(go())
for i, x in enumerate(rows): x["attempt_low"], x["attempt_high"] = v["low"][i], v["high"][i]; x.pop("q")
json.dump(rows, open(f"{R}/nocot/st1_ignore2_iq_rows.json", "w"))
print(len(rows))
