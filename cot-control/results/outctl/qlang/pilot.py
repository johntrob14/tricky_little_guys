"""Question-language pilot: does base gpt-oss-120b reason in the language the question is asked in? 100 seeded questions translated to
French and Spanish by Claude Sonnet 5.5 (translate.py). Conditions: 'format' = translated question + translated ANSWER-format line;
'free' = translated question only. Nothing about reasoning or response language is said. 2 samples each, medium effort, T=1, Tinker.
Language of reasoning and answer by cot-control/rl/controls.lang_profile.  Run from cot-control/rl:  uv run python ../results/outctl/qlang/pilot.py [model_key]"""
import asyncio, json, os, sys, collections
sys.path.insert(0, "."); sys.path.insert(0, "../grading"); sys.path.insert(0, "../results/outctl")
import tinker, controls
from cotswitch import extract
import astra_more as AM
D = os.path.dirname(os.path.abspath(__file__)); MKEY = sys.argv[1] if len(sys.argv) > 1 else "base"
FMT = {"French": "\n\n\nVotre réponse doit respecter le format suivant : 'ANSWER: X' (sans les guillemets), où X est la lettre choisie.",
       "Spanish": "\n\n\nSu respuesta debe tener el siguiente formato: 'ANSWER: X' (sin comillas), donde X es la letra elegida."}
def dom(t):
    p = controls.lang_profile(t); c = collections.Counter(p)
    return (c.most_common(1)[0][0] if c else "none"), len(p), c
async def main():
    sc = tinker.ServiceClient(); path = AM.MODELS[MKEY]
    c = await (sc.create_sampling_client_async(model_path=path) if path else sc.create_sampling_client_async(base_model="openai/gpt-oss-120b"))
    tok = c.get_tokenizer(); p = tinker.SamplingParams(max_tokens=8000, temperature=1.0, top_p=1.0, stop=[200002, 200012]); sem = asyncio.Semaphore(64)
    jobs = [(json.loads(l), cond) for L in FMT for l in open(f"{D}/qs_{L}.jsonl") for cond in ("format", "free")]
    async def one(q, cond):
        user = q["translated"] + (FMT[q["lang"]] if cond == "format" else "")
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
        async with sem: res = (await c.sample_async(prompt=tinker.ModelInput.from_ints(tok.encode(text, add_special_tokens=False)), num_samples=2, sampling_params=p)).sequences
        rows = []
        for k, s in enumerate(res):
            sp = extract.split_output(extract.harmony_to_think(tok.decode(s.tokens, skip_special_tokens=False)), None, "length" if s.stop_reason == "length" else "stop")
            rd, rn, rc = dom(sp["reasoning"]); ad, an, ac = dom(sp["response"])
            pred = extract.parse_answer_lenient(sp["response"])
            rows.append(dict(model=MKEY, key=q["key"], lang=q["lang"], cond=cond, sample=k, user=user, reasoning=sp["reasoning"], response=sp["response"], stop=s.stop_reason,
                             n_tokens=len(s.tokens), r_dominant=rd, r_sentences=rn, r_target_share=(rc[q["lang"]] / rn if rn else None), a_dominant=ad, a_target_share=(ac[q["lang"]] / an if an else None),
                             pred=pred, correct=pred == q["gold"]))
        return rows
    rows = [x for g in await asyncio.gather(*(one(q, cond) for q, cond in jobs)) for x in g]
    with open(f"{D}/pilot_{MKEY}.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in rows)
    print("est $", round(sum(r["n_tokens"] for r in rows) * 0.84e-6 + len(rows) / 2 * 450 * 0.33e-6, 3))
    print("| question language | condition | n | reasoning mostly in that language | reasoning >=95% | mean share | answer mostly in that language | accuracy |")
    for L in FMT:
        for cond in ("format", "free"):
            v = [r for r in rows if r["lang"] == L and r["cond"] == cond]; sh = [r["r_target_share"] or 0 for r in v]
            print(f"| {L} | {cond} | {len(v)} | {sum(r['r_dominant'] == L for r in v)} | {sum((r['r_target_share'] or 0) >= .95 and r['r_sentences'] >= 3 for r in v)} | "
                  f"{sum(sh)/len(sh):.2f} | {sum(r['a_dominant'] == L for r in v)} | {sum(r['correct'] for r in v)/len(v):.2f} |")
asyncio.run(main())
