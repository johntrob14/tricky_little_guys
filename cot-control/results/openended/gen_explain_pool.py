"""Generate easy open-ended explanation questions for ogrpo1 (output-side many-types control). gpt-oss-120b via OpenRouter (DeepInfra),
50 topics x 20 questions, deduplicated. Writes rl/data/explain_pool.jsonl. Run from cot-control/rl."""
import asyncio, json, os, re, sys, random
sys.path.insert(0, ".")
from orjudge import OpenRouterJudge
TOPICS = """photosynthesis; the water cycle; how vaccines work; plate tectonics; why the sky is blue; supply and demand; inflation; how a bill becomes law;
the causes of World War I; the Renaissance; how computers store data; how the internet routes information; recursion in programming; why ice floats;
the greenhouse effect; how the heart pumps blood; the immune system; DNA and heredity; evolution by natural selection; why seasons happen; tides and the moon;
how airplanes fly; electricity and circuits; magnets; sound waves; how rainbows form; compound interest; why diversification reduces investment risk;
the scientific method; correlation versus causation; how memory works in the brain; sleep and why we need it; how muscles grow; nutrition and calories;
how bread rises; why metals conduct heat; recycling and why it matters; how a refrigerator works; how batteries store energy; erosion and rivers;
the French Revolution; the industrial revolution; democracy versus monarchy; how language changes over time; why we have time zones; how maps project the globe;
probability and coin flips; why prime numbers matter; the Pythagorean theorem; why the ocean is salty""".replace("\n", " ").split("; ")
PROMPT = """Write 20 short, easy, open-ended questions about {topic} that a curious high-school student might ask and that are best answered with a clear explanation of a few paragraphs (for example "Why does ...", "How does ...", "What happens when ...", "Explain ..."). No multiple choice, no yes/no questions, no questions needing a single number, no questions that depend on an image or a text not given. Vary the angle. Output only the questions, one per line, no numbering."""
async def main():
    j = OpenRouterJudge("openai/gpt-oss-120b", effort="low", concurrency=25, provider="DeepInfra")
    outs = await asyncio.gather(*(j.ask(PROMPT.format(topic=t.strip())) for t in TOPICS))
    seen, rows = set(), []
    for t, (txt, _, _) in zip(TOPICS, outs):
        for line in txt.splitlines():
            q = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip()
            k = re.sub(r"\W+", " ", q.lower()).strip()
            if 25 <= len(q) <= 220 and q.endswith("?") or (25 <= len(q) <= 220 and q.lower().startswith("explain")):
                if k not in seen: seen.add(k); rows.append(dict(id=len(rows), topic=t.strip(), question=q))
    random.Random(0).shuffle(rows)
    with open("data/explain_pool.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in rows)
    print(len(rows), "questions; cost $", round(j.usd, 3))
asyncio.run(main())
