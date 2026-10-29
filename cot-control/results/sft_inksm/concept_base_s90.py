# concept_suppression base rate on the inkrl1 step-90 sampler: the same 64 questions (seed 5) and settings as concept_base_rate_s70.json, current (v2) requirement.
import asyncio, json, random
import tinker, inkio, grpo_types as gt
from orjudge import TinkerJudge
P=[json.loads(l) for l in open('data/pool_k4_inksm.jsonl')]
P=[q for q in P if gt.eligible_type('concept_suppression', q) and all(q.get(f'base_words_{e}',99)>=10 for e in ('0.3','0.6','0.9'))]
rng=random.Random(5); Q=rng.sample(P, 64)
MP=[c['sampler'] for c in json.load(open('../results/rl/inkrl1-s81b/checkpoints.json')) if c['step']==90][0]
s70={r['qid'] for r in json.load(open('../results/sft_inksm/concept_base_rate_s70.json'))}
async def main():
    cl=await tinker.ServiceClient().create_sampling_client_async(model_path=MP); tok=inkio.Tok()
    gj=TinkerJudge(effort="low", max_tokens=1500); jj=TinkerJudge(effort="high", max_tokens=8000)
    params=tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=8000, stop=[tok.eos_token_id])
    tasks=[]
    for i,q in enumerate(Q):
        t=gt.make_task(q,'concept_suppression',random.Random(i),0.3,0.3,False,None,['reasoning','thinking','scratchpad'])
        t|=dict(effort=0.6 if i<40 else 0.9, base_words=q[f"base_words_{'0.6' if i<40 else '0.9'}"])
        tasks.append(t)
    assert {t['qid'] for t in tasks}==s70, "question set differs from step 70"
    async def samp(t):
        s=(await cl.sample_async(prompt=tinker.ModelInput.from_ints(inkio.render(t['user'],t['system'],t['effort'])), num_samples=2, sampling_params=params)).sequences
        return [gt.score_roll(t, gt.tml_think(tok, x.tokens), x.stop_reason=='length', False, '', '') for x in s]
    scs=await asyncio.gather(*[samp(t) for t in tasks])
    pairs=[(t,sc) for t,ss in zip(tasks,scs) for sc in ss]
    await gt.run_judges(gj, jj, pairs, False)
    rows=[]
    for t,sc in pairs:
        ok=sc['fmt_ok'] and sc['gate'] is True
        rows.append(dict(qid=t['qid'], effort=t['effort'], concept=t['value'], frame=t['frame'], instruction=t['instruction'], q=t['qblock'][:300], cot=sc['reasoning'], response=sc['response'], correct=sc['correct'], fmt_ok=sc['fmt_ok'], gate=sc['gate'], req=sc['req'], req_final=sc.get('req_final'), meta=sc['meta'], truncated=sc['truncated'], credited=bool(ok and sc['req'] and sc['meta'] is True and sc['correct'])))
    json.dump(rows, open('../results/sft_inksm/concept_base_rate_s90.json','w'), indent=1)
    for e in (0.6,0.9):
        R=[r for r in rows if r['effort']==e]
        print(f"effort {e}: n {len(R)} | judge pass {sum(bool(r['req']) for r in R)} | credited {sum(r['credited'] for r in R)} | correct {sum(r['correct'] for r in R)} | mention {sum(r['meta'] is False for r in R)} | groups credited {len({r['qid'] for r in R if r['credited']})}/{len({r['qid'] for r in R})}")
    print('judge usd', round(gj.usd+jj.usd,3))
asyncio.run(main())
