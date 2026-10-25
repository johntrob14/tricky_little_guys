"""Attempt check (grpo.JUDGE_PROMPT, gpt-oss-120b effort low via OpenRouter) on every existing CoT and output sample, ignore_question
excluded. Writes more/attempt.jsonl rows {model, side, dataset, item_idx, constraint, sample_idx, attempt}. Run from cot-control/rl."""
import asyncio, json, re, sys
sys.path.insert(0, ".")
from grpo import JUDGE_PROMPT
from orjudge import OpenRouterJudge
O = "../results/outctl"; R = "../results/tinker-runs"
ONLY = sys.argv[1] if len(sys.argv) > 1 else None  # label one model only, appending to attempt.jsonl
COT = {"base": "gptoss120b-t1", "grpo3": "grpo3-step100-t1", "st1-ignore2": "astra-st1-ignore2-medium-t1", "len2": "len2-step60-t1",
       "efflen1": "efflen1-step100-t1", "st1-third": "astra-st1-third-medium-t1", "st1-ignore3": "astra-st1-ignore3-j20b-medium-t1"}
Q = {(x["dataset"], str(x["item_idx"]), x["constraint"]): x["user_prompt"].split("\n\nRequirement:")[0].split("\n\n\nYour response should")[0]
     for x in map(json.loads, open(f"{R}/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl"))}
jobs = []
for m, run in [(a, b) for a, b in COT.items() if ONLY is None or a == ONLY]:
    for x in map(json.loads, open(f"{R}/{run}/generations/gpt-oss-120b/astra.jsonl")):
        jobs.append((m, "cot", x["dataset"], str(x["item_idx"]), x["constraint"], 0, x["reasoning"] or ""))
    for g in json.load(open(f"{O}/astra_out/{m}/graded.json")):
        jobs.append((m, "out", g["dataset"], str(g["item_idx"]), g["constraint"], 0, g["body"]))
for m in [x for x in list(COT) + ["gsft1"] if ONLY is None or x == ONLY]:
    for side in ("cot", "out"):
        for g in json.load(open(f"{O}/more/{side}/{m}/graded.json")):
            jobs.append((m, side, g["dataset"], str(g["item_idx"]), g["constraint"], g["sample_idx"], g["body"]))
jobs = [j for j in jobs if j[4] != "ignore_question" and (ONLY is None or j[0] == ONLY)]
print(len(jobs), "jobs", file=sys.stderr)
async def main():
    j = OpenRouterJudge("openai/gpt-oss-120b", effort="low", concurrency=96)
    async def one(job):
        if not job[6].strip(): return False
        for _ in range(4):
            f, _, _ = await j.ask(JUDGE_PROMPT.format(question=Q[job[2:5]], reasoning=job[6]))
            mm = re.match(r"\W*(YES|NO)\b", f.strip().upper())
            if mm: return mm.group(1) == "YES"
        return None
    res = await asyncio.gather(*(one(x) for x in jobs))
    with open(f"{O}/more/attempt.jsonl", "a" if ONLY else "w") as f:
        for x, a in zip(jobs, res):
            f.write(json.dumps(dict(model=x[0], side=x[1], dataset=x[2], item_idx=x[3], constraint=x[4], sample_idx=x[5], attempt=a)) + "\n")
    print("judge $", round(j.usd, 2), "unparsed", sum(a is None for a in res), file=sys.stderr)
asyncio.run(main())
