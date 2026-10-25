# st1_attempt.py's verdicts (grpo.JUDGE_PROMPT, gpt-oss-120b, Groq, effort low, medium retry) for one new transfer run, then substantive far passes.
import asyncio, json, sys
sys.path.insert(0, ".")
import grpo, st1_compare as sc
from grpo_types import yes_no
from orjudge import OpenRouterJudge
RUN, OUT = "gptoss-st1-ignore3-j20b", sc.ATTEMPT
latest = {r["key"]: r["attempt"] for r in map(json.loads, open(OUT))}
todo = [(f"transfer|{RUN}|{r['key']}", r["qblock"], r["reasoning"] or "") for r in map(json.loads, open(sc.TRANSFER / RUN / "gens.jsonl"))
        if r["type"] != "baseline" and r["frame"] == "F0" and r["placement"] == "user"]
todo = [t for t in todo if latest.get(t[0]) is None]
async def go(todo, eff):
    j = OpenRouterJudge("openai/gpt-oss-120b", effort=eff, max_tokens=1500 if eff == "low" else 4000, concurrency=64, provider="Groq")
    outs = await asyncio.gather(*[j.ask(grpo.JUDGE_PROMPT.format(question=q, reasoning=t)) if t.strip() else asyncio.sleep(0, ("NO",)) for _, q, t in todo])
    with open(OUT, "a") as f:
        for (k, _, _), (fin, *_) in zip(todo, outs):
            latest[k] = yes_no(fin); f.write(json.dumps(dict(key=k, attempt=latest[k], final=fin[:80])) + "\n")
    return j.usd
usd = asyncio.run(go(todo, "low"))
redo = [t for t in todo if latest[t[0]] is None]
if redo: usd += asyncio.run(go(redo, "medium"))
print("judged", len(todo), "retried", len(redo), f"${usd:.2f}")
df = sc.transfer_rows("st1-ignore3-j20b", [RUN], latest)
far = df[df.constraint.isin(sc.COMMON_FAR)]
print("far rows", len(far), "strict", int(far.strict.sum()), "substantive", int((far.strict & (far.attempt == True)).sum()), "unparsed", int((far.attempt.isna()).sum()))
for t, g in far.groupby("constraint"): print(f"  {t}: strict {int(g.strict.sum())}, substantive {int((g.strict & (g.attempt == True)).sum())}")
