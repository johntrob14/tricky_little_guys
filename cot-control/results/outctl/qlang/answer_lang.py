"""Does a checkpoint answer (final channel) in the question's language when no answer format is imposed? Translated question only, no
ANSWER-format line, nothing about language. First 50 seeded questions, French/Spanish/Italian, 1 sample, medium effort, T=1.
Run from cot-control/rl:  uv run python ../results/outctl/qlang/answer_lang.py base|tinker://sampler [...]  -> answer_lang.jsonl (appended)"""
import asyncio, json, os, sys, collections
sys.path.insert(0, "."); sys.path.insert(0, "../grading")
import tinker, controls
from cotswitch import extract
D = os.path.dirname(os.path.abspath(__file__)); LANGS = ["French", "Spanish", "Italian"]
QS = [json.loads(l) for L in LANGS for l in list(open(f"{D}/qs_{L}.jsonl"))[:50]]
def dom(t):
    c = collections.Counter(controls.lang_profile(t)); return c.most_common(1)[0][0] if c else "none"
async def run(sc, m):
    c = await (sc.create_sampling_client_async(base_model="openai/gpt-oss-120b") if m == "base" else sc.create_sampling_client_async(model_path=m)); tok = c.get_tokenizer()
    p = tinker.SamplingParams(max_tokens=8000, temperature=1.0, top_p=1.0, stop=[200002, 200012]); sem = asyncio.Semaphore(48)
    async def one(q):
        text = tok.apply_chat_template([{"role": "user", "content": q["translated"]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
        async with sem: s = (await c.sample_async(prompt=tinker.ModelInput.from_ints(tok.encode(text, add_special_tokens=False)), num_samples=1, sampling_params=p)).sequences[0]
        sp = extract.split_output(extract.harmony_to_think(tok.decode(s.tokens, skip_special_tokens=False)), None, "length" if s.stop_reason == "length" else "stop")
        return dict(model=m, key=q["key"], lang=q["lang"], reasoning=sp["reasoning"], response=sp["response"], stop=s.stop_reason, n_tokens=len(s.tokens),
                    r_strict=controls._language(sp["reasoning"], q["lang"], None)[1], r_dom=dom(sp["reasoning"]), a_dom=dom(sp["response"]))
    return await asyncio.gather(*(one(q) for q in QS))
async def main():
    sc = tinker.ServiceClient(); rows = [r for g in await asyncio.gather(*(run(sc, m) for m in sys.argv[1:])) for r in g]
    with open(f"{D}/answer_lang.jsonl", "a") as f: f.writelines(json.dumps(r) + "\n" for r in rows)
    print("est $", round(sum(r["n_tokens"] for r in rows) * 0.84e-6 + len(rows) * 450 * 0.33e-6, 2))
    print("| model | lang | n | reasoning strict | reasoning mostly target | answer mostly target | cap-outs |")
    for m in sys.argv[1:]:
        for L in LANGS:
            x = [r for r in rows if r["model"] == m and r["lang"] == L]
            print(f"| {m.split('/')[-1]} | {L} | {len(x)} | {sum(r['r_strict'] for r in x)} | {sum(r['r_dom'] == L for r in x)} | {sum(r['a_dom'] == L for r in x)} | {sum(r['stop'] == 'length' for r in x)} |")
asyncio.run(main())
