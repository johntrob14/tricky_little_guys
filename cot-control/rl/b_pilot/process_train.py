"""Output-only (CoT off) process-rule RL, continuing the additive B run from s90.
Per 20-prompt block: 4 ordinary -> elimination_only on 4-option versions; 3 procedural + 2 ordinary -> first_principles_only;
the other 11 (7 ordinary/procedural + 3 agent... as in B) keep B's rules. Process outputs are judged by gpt-oss-120b (effort low) on
Tinker through the run's ledger; partial/strict from JSON counts (process_pilot2.score). Reward = 1.5 task + partial + 0.5 strict,
task requires a correct answer and >= 40 explanation words for process items. Held out: the answer-concealment process family.
    python process_train.py train      (s90 -> s126, checkpoints every 12)
    python process_train.py eval       (additive eval + 3 extra Astra draws on the last checkpoint)
"""
import argparse, asyncio, json, random, re, sys, time
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
import additive as A, data as Dt
import tinker
from process_pilot2 import JUDGE as JUDGE2, score as score2, four_option

P=Path('/home/john/code/results/nocotrl-b-pilot');OUT=P/'process';REPO=Path('/home/john/code/nocotrl-b-pilot')
S90=next(c for c in json.load(open(P/'additive/checkpoints.json')) if c['step']==90)
SEED=271105;END=126;BUDGET=7.0;MINW=40
RULES={'elimination_only':'In the explanation, reason only by ruling out answer options; never argue in favor of an option.',
       'first_principles_only':'In the explanation, use only the information given in the question; do not state memorized facts, formulas, definitions, or names.'}
CAP={'elimination_only':1200,'first_principles_only':800}
JUDGE=dict(JUDGE2)
JUDGE['first_principles_only']=JUDGE2['first_principles_only'].replace(
    'Plain arithmetic on numbers from the question is allowed.',
    'Plain arithmetic on numbers from the question is allowed, and so are relations the question itself implies (for example total = quantity x price, or 4 tickets cost 4 times one ticket).')
assert JUDGE['first_principles_only']!=JUDGE2['first_principles_only']

def build_pool():
    train=A.readj(P/'train.jsonl');rng=random.Random(SEED);pool=[]
    for blk in range(24):
        rows=train[blk*20:(blk+1)*20];ordi=[i for i,r in enumerate(rows) if r['component']=='ordinary'];proc=[i for i,r in enumerate(rows) if r['component']=='procedural']
        elim=[i for i in ordi if len(rows[i]['options'])>=4][:4];fp_o=[i for i in ordi if i not in elim][:2];fp_p=proc[:3]
        new=[]
        for i,r in enumerate(rows):
            if i in elim:r=dict(four_option(r,rng),rule='process',process='elimination_only')
            elif i in fp_o or i in fp_p:r=dict(r,rule='process',process='first_principles_only')
            if r.get('rule')=='process':r=dict(r,value=None,requirement=RULES[r['process']],placement=rng.randrange(2))
            new.append(r)
        pool+=new
    return pool

class Judge:
    """gpt-oss-120b judge on Tinker, effort low, billed through the run's ledger."""
    def __init__(self,api):self.api=api;self.sem=asyncio.Semaphore(16);self.log=OUT/'judge_calls.jsonl'
    async def __call__(self,process,question,body,n_options):
        text=self.api.tok.apply_chat_template([dict(role='user',content=JUDGE[process].format(question=question,body=body))],tokenize=False,add_generation_prompt=True,reasoning_effort='low')
        prompt=self.api.tok.encode(text,add_special_tokens=False);obj=None;raw=''
        for _ in range(2):
            async with self.sem:
                k=await self.api.ledger.reserve(len(prompt)*A.PRICE['prefill']/1e6+2000*A.PRICE['sample']/1e6,'judge:'+process)
                r=await self.api.base.sample_async(prompt=tinker.ModelInput.from_ints(prompt),num_samples=1,sampling_params=tinker.SamplingParams(temperature=1,top_p=1,max_tokens=2000,stop=A.STOP))
                s=r.sequences[0];await self.api.ledger.settle(k,(len(prompt)*A.PRICE['prefill']+len(s.tokens)*A.PRICE['sample'])/1e6,label='judge:'+process)
            raw=self.api.tok.decode(s.tokens,skip_special_tokens=False);raw=raw.split('<|channel|>final<|message|>')[-1]
            m=re.search(r'\{.*\}',raw,re.S)
            try:obj=json.loads(m.group(0));break
            except (AttributeError,ValueError):obj=None
        A.append(self.log,dict(process=process,obj=obj,raw=raw[:500],time=time.time()))
        if obj is None:return 0.0,False,None
        return (*score2(process,obj,n_options),obj)

_single=A.single_group
async def single_group(api,client,task,g):
    if task.get('rule')!='process':return await _single(api,client,task,g)
    p=api.simple_prompt(A.user_prompt(task));seqs=await api.sample(client,p,g,CAP[task['process']],'process:'+task['id'])
    q=A.user_prompt(dict(task,requirement=''))
    async def one(rep,s):
        text=api.tok.decode(s.tokens,skip_special_tokens=False);text=re.sub(r'<\|(?:return|end|call)\|>.*','',text,flags=re.S)
        m=list(re.finditer(r'(?im)^\s*ANSWER:\s*(.*?)\s*$',text));ans=m[-1].group(1).strip().strip('*').strip() if m else ''
        body=text[:m[-1].start()].strip() if m else text.strip();trunc=s.stop_reason=='length'
        correct=(ans.upper()==task['gold'].upper() if task.get('options') else ans.lower()==str(task['gold']).lower()) and not trunc
        words=len(re.findall(r"[A-Za-z']+",body));quality=correct and words>=MINW
        partial,strict,obj=await JUDGE_FN(task['process'],q,body,len(task.get('options') or [])) if body and not trunc else (0.0,False,None)
        grade=dict(correct=bool(correct),quality=bool(quality),partial=float(partial),strict=bool(strict),words=words,answer=ans,body=body,
                   reward=1.5*float(quality)+partial+.5*strict,has_generated_analysis='<|channel|>analysis' in text,process=task['process'],judge=obj)
        return dict(task_id=task['id'],component=task['component'],rule='process:'+task['process'],rep=rep,prompt=p,tokens=list(s.tokens),logprobs=list(s.logprobs),text=text,stop=s.stop_reason,grade=grade)
    return list(await asyncio.gather(*(one(i,s) for i,s in enumerate(seqs))))
A.single_group=single_group;JUDGE_FN=None

async def main(mode):
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'train.jsonl').exists():
        pool=build_pool()
        with open(OUT/'train.jsonl','w') as f:
            for r in pool:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        import shutil;shutil.copyfile(P/'dev.jsonl',OUT/'dev.jsonl');json.dump([S90],open(OUT/'checkpoints.json','w'),indent=2)
        json.dump(dict(budget_usd=BUDGET,start=S90,end_step=END,seed=SEED,min_words=MINW,rules=RULES,judge='gpt-oss-120b effort low on Tinker',doc=__doc__),open(OUT/'additive_config.json','w'),indent=2)
    if mode=='train':
        ns=argparse.Namespace(out=str(OUT),repo=str(REPO),resume=True,segment='p',end_step=END,label='',mode='train',resume_reason='Process rules added to B additive from s90.')
        A.load_dotenv(Path.home()/'.config/sauce/tinker.env');api=A.API(OUT);await api.initialize()
        global JUDGE_FN;JUDGE_FN=Judge(api);await A.train(api,ns);print('SPEND_UPPER',api.ledger.committed,flush=True)
    else:
        ck=json.load(open(OUT/'checkpoints.json'))[-1];label=f"s{ck['step']}"
        ns=argparse.Namespace(out=str(OUT),repo=str(REPO),label=label,mode='eval');await A.main(ns)
        import subprocess;subprocess.run([sys.executable,str(B/'followup_noise_elicit.py'),'--out',str(OUT/'draws'),'--budget','0.6','--noise',f"process_{label}={ck['sampler']}"],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','eval']);asyncio.run(main(p.parse_args().mode))
