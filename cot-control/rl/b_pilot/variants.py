"""Small diagnostic: is B's plateau from repeating the same 480 tasks, or from the rules being too easy?
Two 12-step continuations from the additive s90 state (steps 91-102 use train blocks 18-23 and 0-5, 240 prompts):
  fresh:   240 new tasks (144 unused MMLU-Pro + 60 procedural + 36 agent), B's single rules, same per-step mix.
  stacked: the same 240 tasks those steps saw before, each with 2-3 of B's rules stacked (one structural rule plus
           one or two of lower/avoid3/length; lower never with sections). Strict = all rules pass; partial = mean.
Everything else (reward 1.5 task + partial + 0.5 strict, lr, group size, sampler) is additive.py unchanged.
Evaluation: additive.py eval (dev + one Astra draw) and 3 extra Astra draws via followup_noise_elicit.py.
    python variants.py ARM      (ARM in fresh, stacked)
"""
import argparse, asyncio, json, random, re, shutil, subprocess, sys
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
import additive as A
import data as Dt

P=Path('/home/john/code/results/nocotrl-b-pilot');REPO=Path('/home/john/code/nocotrl-b-pilot')
S90=next(c for c in json.load(open(P/'additive/checkpoints.json')) if c['step']==90)
BLOCKS=[18,19,20,21,22,23,0,1,2,3,4,5]
SEED=271101
STRUCT=['json','bullets','count','sections'];MODS=['lower','avoid3','length']

def phrase(rule,value,wording):
    return Dt.PHRASES[rule][wording].format(v=', '.join(value) if rule=='avoid3' else f'{value[0]} to {value[1]}' if rule=='length' else '')

def avoid3_words(r):
    # Same keyword choice as data.assign_rule, independent of the rule draw.
    kw=[x.lower() for x in r.get('keywords',[]) if re.fullmatch('[A-Za-z]{4,}',x)]
    if len(kw)<3:
        common={'which','these','following','would','should','could','there','their','about','answer','question','explain','given','what','with','from','that','this'}
        import collections
        kw+=[w for w,n in collections.Counter(re.findall('[a-z]{4,}',r['question'].lower())).most_common() if w not in common and w not in kw]
    kw=list(dict.fromkeys(kw))
    for x in ['therefore','because','result']:
        if x not in kw:kw.append(x)
    return kw[:3]

def stack_task(r,rng):
    s=rng.choice(STRUCT);mods=[m for m in MODS if not (m=='lower' and s=='sections')]
    chosen=[s]+rng.sample(mods,rng.choice([1,2]));rules=[]
    for rule in chosen:
        value=avoid3_words(r) if rule=='avoid3' else rng.choice([[35,55],[60,90],[100,130]]) if rule=='length' else None
        rules.append(dict(rule=rule,value=value,wording=rng.randrange(3)))
    return dict({k:v for k,v in r.items()},rule='stack',rules=rules,value=None,wording=None,
                requirement=' '.join(phrase(x['rule'],x['value'],x['wording']) for x in rules),placement=rng.randrange(2))

_grade=A.grade
def grade(task,text,trunc=False):
    if task.get('rule')!='stack':return _grade(task,text,trunc)
    parts=[_grade(dict(task,rule=x['rule'],value=x['value']),text,trunc) for x in task['rules']]
    g=dict(parts[0]);g['partial']=sum(p['partial'] for p in parts)/len(parts);g['strict']=all(p['strict'] for p in parts)
    g['per_rule']=[dict(rule=x['rule'],partial=p['partial'],strict=p['strict']) for x,p in zip(task['rules'],parts)]
    g['reward']=1.5*float(g['quality'])+g['partial']+.5*g['strict'];return g
A.grade=grade

def fresh_tasks(used_ids,seed=SEED,idbase=5000,seen=None):
    import hashlib
    from datasets import Dataset
    astra=[json.loads(l) for l in open(REPO/'cot-control/results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl')]
    banned=[x['user_prompt'].split('\n\nRequirement:')[0].split("\n\n\nYour response")[0].removeprefix('Question: ').split('\n\nA)')[0] for x in astra]
    extra=REPO/'cot-control/results/outctl/new_items.json'
    if extra.exists():banned+=[x['user_prompt'].split('\n\nRequirement:')[0].split('\n\nA)')[0].removeprefix('Question: ') for x in json.load(open(extra))]
    ban=[set(Dt.norm(q).split()) for q in banned];banexact={Dt.norm(q) for q in banned}
    arrow=list(Path('/tmp/john-scratch/huggingface/datasets/TIGER-Lab___mmlu-pro/default/0.0.0').glob('*/mmlu-pro-test.arrow'))[0]
    src=list(Dataset.from_file(str(arrow)));rng=random.Random(seed);rng.shuffle(src);out=[]
    for x in src:
        if f'mmlu-{x["question_id"]}' in used_ids or len(x['options'])<2 or len(x['question'])>4000:continue
        n=Dt.norm(x['question']);w=set(n.split())
        if n in banexact or any(len(w&b)/max(1,len(w|b))>=.5 for b in ban):continue
        out.append(dict(id=f'mmlu-{x["question_id"]}',component='ordinary',family=x['category'],question=x['question'],options=x['options'],gold=x['answer'],keywords=[],old_pool=False))
        if len(out)==144:break
    rows=[];seen=set(seen or ())
    def gen(f,i):  # redraw on a repeated question, as data.py's deduplicate does
        r=f(rng,i,'fresh')
        while Dt.fingerprint(r['question']) in seen:r=f(rng,i,'fresh')
        seen.add(Dt.fingerprint(r['question']));return r
    for blk in range(12):
        batch=out[blk*12:(blk+1)*12]+[gen(Dt.procedural,idbase+blk*5+j) for j in range(5)]+[gen(Dt.agent,idbase+blk*3+j) for j in range(3)]
        for j,r in enumerate(batch):rows.append(Dt.assign_rule(r,rng,blk*20+j))
    return rows

def build(arm,out):
    train=A.readj(P/'train.jsonl');assert len(train)==480
    if arm=='fresh':
        used={r['id'] for r in train}|{r['id'] for r in A.readj(P/'dev.jsonl')}
        new=fresh_tasks(used);seen={Dt.fingerprint(r['question']) for r in train}
        assert not any(Dt.fingerprint(r['question']) in seen for r in new if r['component']=='ordinary')
    else:
        rng=random.Random(SEED);new=[stack_task(r,rng) for b in BLOCKS for r in train[b*20:(b+1)*20]]
    pool=list(train)
    for i,b in enumerate(BLOCKS):pool[b*20:(b+1)*20]=new[i*20:(i+1)*20]
    with open(out/'train.jsonl','w') as f:
        for r in pool:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    shutil.copyfile(P/'dev.jsonl',out/'dev.jsonl')
    json.dump([S90],open(out/'checkpoints.json','w'),indent=2)
    json.dump(dict(budget_usd=3.0,arm=arm,seed=SEED,start=S90,steps='91-102',blocks=BLOCKS,doc=__doc__),open(out/'additive_config.json','w'),indent=2)

BLOCKS2=list(range(6,18))  # steps 103-114 read blocks 6-17, still original tasks after segment 1

def build_segment2(arm,out):
    """Replace blocks 6-17 with new tasks of the arm's kind; segment-1 pool kept as train_seg1.jsonl."""
    shutil.copyfile(out/'train.jsonl',out/'train_seg1.jsonl')
    pool=A.readj(out/'train.jsonl');orig=A.readj(P/'train.jsonl')
    if arm=='fresh':
        used={r['id'] for r in pool}|{r['id'] for r in orig}|{r['id'] for r in A.readj(P/'dev.jsonl')}
        seen={Dt.fingerprint(r['question']) for r in pool+orig}
        new=fresh_tasks(used,seed=SEED+1,idbase=6000,seen=seen)
        assert not any(Dt.fingerprint(r['question']) in seen for r in new),'segment-2 fresh task repeats an earlier question'
    else:
        rng=random.Random(SEED+1);new=[stack_task(r,rng) for b in BLOCKS2 for r in orig[b*20:(b+1)*20]]
    for i,b in enumerate(BLOCKS2):pool[b*20:(b+1)*20]=new[i*20:(i+1)*20]
    with open(out/'train.jsonl','w') as f:
        for r in pool:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    cfg=json.load(open(out/'additive_config.json'));cfg.update(budget_usd=6.0,segment2=dict(steps='103-114',blocks=BLOCKS2,seed=SEED+1,note='John asked for 12 more steps on both arms'))
    json.dump(cfg,open(out/'additive_config.json','w'),indent=2)

async def run2(arm):
    out=P/'variants'/arm
    if not (out/'train_seg1.jsonl').exists():build_segment2(arm,out)
    ns=argparse.Namespace(out=str(out),repo=str(REPO),resume=True,segment='v2',end_step=114,label='s114',resume_reason=f'Variant {arm}: 12 more steps from s102.')
    if not (out/'steps_v2.jsonl').exists():ns.mode='train';await A.main(ns)
    ns.mode='eval';await A.main(ns)
    sp=json.load(open(out/'checkpoints.json'))[-1]['sampler']
    subprocess.run([sys.executable,str(B/'followup_noise_elicit.py'),'--out',str(out/'draws2'),'--budget','0.6','--noise',f'{arm}_s114={sp}'],check=True)

async def run(arm):
    out=P/'variants'/arm;out.mkdir(parents=True,exist_ok=True)
    if not (out/'train.jsonl').exists():build(arm,out)
    ns=argparse.Namespace(out=str(out),repo=str(REPO),resume=True,segment='v',end_step=102,label='s102',resume_reason=f'Variant {arm} from additive s90.')
    if not (out/'steps_v.jsonl').exists():ns.mode='train';await A.main(ns)
    ns.mode='eval';await A.main(ns)
    sp=json.load(open(out/'checkpoints.json'))[-1]['sampler']
    subprocess.run([sys.executable,str(B/'followup_noise_elicit.py'),'--out',str(out/'draws'),'--budget','0.6','--noise',f'{arm}_s102={sp}'],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['fresh','stacked']);p.add_argument('--segment2',action='store_true');a=p.parse_args()
    asyncio.run(run2(a.arm) if a.segment2 else run(a.arm))
