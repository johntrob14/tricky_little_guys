"""Add 'keywords' (3 terms a natural answer would use) to each explain-pool question, for the word-suppression constraints. gpt-oss-120b
via OpenRouter (DeepInfra). Rewrites rl/data/explain_pool.jsonl in place (keeps questions with 3 single-word keywords). Run from cot-control/rl."""
import asyncio, json, re, sys
sys.path.insert(0, ".")
from orjudge import OpenRouterJudge
P = "data/explain_pool.jsonl"
rows = [json.loads(l) for l in open(P)]
PROMPT = """Question: {q}

List the 3 single words (no phrases) that a clear explanation answering this question would be most likely to use repeatedly. Choose content words specific to the topic, not generic words like "because" or "process". Output only the 3 words, comma-separated, lowercase."""
async def main():
    j = OpenRouterJudge("openai/gpt-oss-120b", effort="low", concurrency=64, provider="DeepInfra")
    outs = await asyncio.gather(*(j.ask(PROMPT.format(q=r["question"])) for r in rows))
    keep = []
    for r, (txt, _, _) in zip(rows, outs):
        ws = [w.strip().strip(".").lower() for w in txt.strip().splitlines()[-1].split(",")]
        ws = [w for w in ws if re.fullmatch(r"[a-z][a-z\-]{2,}", w)]
        if len(ws) >= 3: r["keywords"] = ws[:3]; keep.append(r)
    with open(P, "w") as f: f.writelines(json.dumps(r) + "\n" for r in keep)
    print(len(keep), "of", len(rows), "kept; cost $", round(j.usd, 3))
asyncio.run(main())
