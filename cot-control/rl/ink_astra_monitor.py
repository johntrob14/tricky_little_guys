"""Astra 108 with the CoT on, at every saved checkpoint of the Inkling-Small nocotrl runs (and base first): thinking effort 0.6, 16k cap,
4 samples per question, reference graders on the 96 rule-graded items, paired against base. Monitoring only (no stopping rule).
ignore_question rows are stored for judging later.
    python ink_astra_monitor.py RUN_DIR [RUN_DIR ...]       (polls each RUN_DIR/checkpoints.json until both runs are done)
"""
import asyncio, json, sys, time
from pathlib import Path
import numpy as np
import tinker
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parent/'grading')]
import inkio
from cotswitch import extract, graders as G
from astra_continuous import score

import os
TERM=os.environ.get('INK_TERM');SUF=f'__{TERM}' if TERM else ''
MODEL='thinkingmachines/Inkling-Small';EFFORT=0.6;CAP=16000;N=4;PRICE=dict(prefill=0.58,sample=1.44);BUDGET=12.0
ASTRA=HERE.parent/'results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl'
OUT=Path('/home/john/code/results/nocotrl-ink/astra_monitor');OUT.mkdir(parents=True,exist_ok=True)
tok=inkio.Tok();spent=[0.0]

def split(gen):
    try:t=inkio.to_think(gen)
    except AssertionError:return dict(reasoning='',response='',malformed=True)
    sp=extract.split_output(t,None,'stop');return dict(reasoning=sp['reasoning'],response=sp['response'],content=sp['content'],malformed=False)

async def run(svc,label,model_path,items,sem):
    path=OUT/f'{label}.jsonl'
    done={(r['key'],r['sample']) for r in map(json.loads,open(path))} if path.exists() else set()
    client=await (svc.create_sampling_client_async(base_model=MODEL) if model_path=='base' else svc.create_sampling_client_async(model_path=model_path))
    params=tinker.SamplingParams(temperature=1.0,top_p=1.0,max_tokens=CAP,stop=[tok.eos_token_id])
    async def one(t):
        if all((t['key'],i) in done for i in range(N)):return
        if spent[0]>BUDGET:return
        p=inkio.render(t['user_prompt'],effort=EFFORT)
        async with sem:r=await client.sample_async(prompt=tinker.ModelInput.from_ints(p),num_samples=N,sampling_params=params)
        spent[0]+=(len(p)*N*PRICE['prefill']+sum(len(s.tokens) for s in r.sequences)*PRICE['sample'])/1e6
        with open(path,'a') as f:
            for i,s in enumerate(r.sequences):
                sp=split(tok.decode(list(s.tokens)))
                correct=extract.parse_answer(sp['response'],sp.get('content',sp['response']))==t['answer_gold'] if sp['response'] else False
                f.write(json.dumps(dict(key=t['key'],sample=i,constraint=t['constraint'],grader_args=t['grader_args'],cot=sp['reasoning'],response=sp['response'],
                                        malformed=sp['malformed'],stop=s.stop_reason,n_tokens=len(s.tokens),correct=correct))+'\n')
    await asyncio.gather(*(one(t) for t in items))
    return [json.loads(l) for l in open(path)]

def graded(rows):
    d={}
    for r in rows:
        if r['constraint']=='ignore_question':continue
        s=G.grade('astra',r['constraint'],r['cot'],r['grader_args'])['strict'] if r['cot'] else 0
        d.setdefault(r['key'],[]).append((float(s or 0),score(dict(constraint=r['constraint'],reasoning=r['cot'],grader_args=r['grader_args']),s) if r['cot'] else 0.0,len(r['cot'].split())))
    return d

def summary(label,rows,base,rng=np.random.default_rng(8)):
    g=graded(rows);out=dict(label=label,n=sum(len(v) for v in g.values()),strict=float(np.mean([x[0] for v in g.values() for x in v])),partial=float(np.mean([x[1] for v in g.values() for x in v])),
        median_cot_words=float(np.median([x[2] for v in g.values() for x in v])),trunc=sum(r['stop']=='length' for r in rows),malformed=sum(r['malformed'] for r in rows),
        correct=float(np.mean([r['correct'] for r in rows])),spent_total=round(spent[0],3))
    if base is not None and label!='base':
        K=sorted(set(g)&set(base))
        for i,m in ((0,'strict'),(1,'partial')):
            x=np.array([np.mean([t[i] for t in g[k]])-np.mean([t[i] for t in base[k]]) for k in K]);bs=rng.choice(x,(10000,len(x))).mean(1)
            out[f'd_{m}']=[round(float(x.mean()),4),round(float(np.quantile(bs,.025)),4),round(float(np.quantile(bs,.975)),4)]
    return out

async def main(runs):
    for l in open(Path.home()/'.config/sauce/tinker.env'):
        k,_,v=l.strip().removeprefix('export ').partition('=')
        if k and not k.startswith('#'):__import__('os').environ.setdefault(k,v.strip('"\''))
    svc=tinker.ServiceClient();sem=asyncio.Semaphore(16);items=[json.loads(l) for l in open(ASTRA)]
    if TERM:items=[dict(t,user_prompt=t['user_prompt'].replace('analysis channel',TERM)) for t in items]
    log=OUT/f'summary{SUF}_monitor.jsonl' if SUF else OUT/'summary.jsonl';seen={json.loads(l)['label'] for l in open(log)} if log.exists() else set()
    base_rows=await run(svc,'base'+SUF,'base',items,sem);base=graded(base_rows)
    if 'base' not in seen:
        s=summary('base',base_rows,None);open(log,'a').write(json.dumps(s)+'\n');print(json.dumps(s),flush=True);seen.add('base')
    while True:
        for rd in runs:
            ck=Path(rd)/'checkpoints.json'
            for c in (json.load(open(ck)) if ck.exists() else []):
                label=f"{Path(rd).name}_s{c['step']}{SUF}"
                if label in seen:continue
                s=summary(label,await run(svc,label,c['sampler'],items,sem),base);open(log,'a').write(json.dumps(s)+'\n');print(json.dumps(s),flush=True);seen.add(label)
        finished=all('done.' in (Path(str(rd)+'.log').read_text() if Path(str(rd)+'.log').exists() else '') or 'budget' in (Path(str(rd)+'.log').read_text() if Path(str(rd)+'.log').exists() else '') for rd in runs)
        pending=[1 for rd in runs for c in (json.load(open(Path(rd)/'checkpoints.json')) if (Path(rd)/'checkpoints.json').exists() else []) if f"{Path(rd).name}_s{c['step']}{SUF}" not in seen]
        if finished and not pending:print('MONITOR_DONE',flush=True);return
        if spent[0]>BUDGET:print('MONITOR_BUDGET',flush=True);return
        await asyncio.sleep(60)

if __name__=='__main__':asyncio.run(main(sys.argv[1:]))
