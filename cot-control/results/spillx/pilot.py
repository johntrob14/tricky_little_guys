"""Base-rate pilot for spillx behaviors: base gpt-oss-120b (default chat template), each behavior alone (12 prompts x 4) and stacks of 2 and 3
(24 prompts x 4 each), T=1, max 2000 tokens. Run from cot-control/rl."""
import asyncio, json, random, sys, collections, os
sys.path.insert(0, "."); sys.path.insert(0, "../grading")
import tinker, outstyles as S, spillx_tasks as X
from cotswitch import extract
O = os.path.dirname(os.path.abspath(__file__))
async def main():
    c = await tinker.ServiceClient().create_sampling_client_async(base_model="openai/gpt-oss-120b"); tok = c.get_tokenizer()
    p = tinker.SamplingParams(max_tokens=2000, temperature=1.0, top_p=1.0, stop=[200002, 200012]); sem = asyncio.Semaphore(64); rng = random.Random(0)
    tasks = [dict(X.make(rng, [b], 1), cond=f"single:{b}") for b in S.B for _ in range(12)] + [dict(X.make(rng, S.TRAIN, k), cond=f"stack{k}") for k in (2, 3) for _ in range(24)]
    async def one(t):
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": t["user"]}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)
        async with sem: res = (await c.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=4, sampling_params=p)).sequences
        out = []
        for s in res:
            sp = extract.split_output(extract.harmony_to_think(tok.decode(s.tokens, skip_special_tokens=False)), None, "length" if s.stop_reason == "length" else "stop")
            g = S.grade(t["nv"], sp["response"]); cg = S.grade(t["nv"], sp["reasoning"])
            pred = extract.parse_answer_lenient(sp["response"]) if t["kind"] == "mc" else None
            out.append(dict(cond=t["cond"], kind=t["kind"], nv=t["nv"], user=t["user"], reasoning=sp["reasoning"], response=sp["response"], n_tokens=len(s.tokens), trunc=s.stop_reason == "length",
                            per=g, cot_per=cg, partial=sum(x for x, _ in g) / len(g), all_strict=all(s_ for _, s_ in g), correct=(pred == t["gold"]) if t["kind"] == "mc" else None))
        return out
    rows = [x for g in await asyncio.gather(*(one(t) for t in tasks)) for x in g]
    with open(f"{O}/pilot_base.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in rows)
    print("est $", round(sum(r["n_tokens"] for r in rows) * 0.84e-6 + len(rows) / 4 * 600 * 0.33e-6, 2))
    print("| condition | n | strict | partial | CoT strict (same rule, unasked) | MC acc | trunc |")
    for cond in sorted({r["cond"] for r in rows}, key=lambda c: (c.startswith("stack"), c)):
        x = [r for r in rows if r["cond"] == cond]; mc = [r for r in x if r["kind"] == "mc"]
        print(f"| {cond} | {len(x)} | {sum(r['all_strict'] for r in x)/len(x):.2f} | {sum(r['partial'] for r in x)/len(x):.2f} | {sum(all(s for _, s in r['cot_per']) for r in x)/len(x):.2f} | {(sum(r['correct'] for r in mc)/len(mc)) if mc else float('nan'):.2f} | {sum(r['trunc'] for r in x)} |")
asyncio.run(main())
