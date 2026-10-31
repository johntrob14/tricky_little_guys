import json,re,sys,hashlib,random,statistics
from pathlib import Path
import numpy as np
C=Path('/home/john/code/tlg-inksm/cot-control');D=Path('/home/john/code/tlg-arena-rl/cot-control');O=Path('outputs');sys.path.insert(0,str(C/'grading'))
from cotswitch import graders as G
rng=np.random.default_rng(20261030);hashes={};raw={};data={'astra':{},'deepseek':{}}
def read(p):
 b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();return [json.loads(l) for l in b.splitlines() if l.strip()]
def norm(s): return re.sub(r'[^a-z0-9]','',s.lower())
def wilson(k,n):
 z=1.95996398454;p=k/n;mid=(p+z*z/(2*n))/(1+z*z/n);h=z/(1+z*z/n)*np.sqrt(p*(1-p)/n+z*z/(4*n*n));return [float(mid-h),float(mid+h)]
runs=['qwen38-run2','opsd1-step30-t1','opsd-grpo1-step30-t1','inksmtk-up-e06','inksft1-ep2-req-e06','inkrl1-s80-req-e06','inksmtk-up','inksft1-ep2-req-e09','inkrl1-s80-req-e09']
for run in runs:
 ps=list((C/'results/tinker-runs'/run/'generations').glob('*/astra.jsonl'));assert len(ps)==1
 rs=[r for r in read(ps[0]) if r['constraint'] not in ['ignore_question','baseline']];raw[run]=rs;out=[]
 for r in rs:
  strict=bool(G.grade('astra',r['constraint'],r['reasoning'] or '',r['grader_args'])['strict']);a=re.search(r'ANSWER\s*:\s*([A-Z])\b',r['response'] or '',re.I);correct=bool(a and a[1].upper()==r['answer_gold']);assert correct==r['correct'];valid=bool((r['reasoning'] or '').strip() and r.get('think_closed') and r.get('finish_reason')!='length' and a)
  out.append({'key':r['key'],'question':norm(r['user_prompt'].split('\n\nA)')[0]),'strict':int(strict),'valid':int(valid),'correct':int(correct),'joint':int(strict and valid and correct),'words':len((r['reasoning'] or '').split())})
 assert len(out)==96 and len({r['key'] for r in out})==96
 raw[run+'_graded']=out
 qs=sorted({x['question'] for x in out});sums=np.array([sum(x['strict'] for x in out if x['question']==q) for q in qs]);ns=np.array([sum(x['question']==q for x in out) for q in qs]);idx=rng.integers(len(qs),size=(10000,len(qs)));boots=sums[idx].sum(1)/ns[idx].sum(1);ci=np.quantile(boots,[.025,.975]).tolist()
 if sum(sums)==0:ci=[0,wilson(0,len(qs))[1]]
 data['astra'][run]={'n':96,'question_clusters':len(qs),'strict':sum(x['strict'] for x in out),'valid':sum(x['valid'] for x in out),'correct':sum(x['correct'] for x in out),'joint':sum(x['joint'] for x in out),'strict_valid':sum(x['strict'] and x['valid'] for x in out),'strict_ci95':ci,'median_words':statistics.median(x['words'] for x in out),'sampling':rs[0]['sampling']}
for run,base in [(r,'qwen38-run2' if r.startswith('opsd') else 'inksmtk-up-e06') for r in runs if r.startswith('opsd') or r in ['inksft1-ep2-req-e06','inkrl1-s80-req-e06']]:
 b={x['key']:x for x in raw[base+'_graded']};xs=raw[run+'_graded'];qs=sorted({x['question'] for x in xs});ns=np.array([sum(x['question']==q for x in xs) for q in qs]);sums=np.array([sum(x['strict']-b[x['key']]['strict'] for x in xs if x['question']==q) for q in qs]);idx=rng.integers(len(qs),size=(10000,len(qs)));data['astra'][run]['delta']=float(sum(sums)/sum(ns));data['astra'][run]['delta_ci95']=np.quantile(sums[idx].sum(1)/ns[idx].sum(1),[.025,.975]).tolist()
pre=read(D/'results/obf_eval/obf2g-s25-e25/rows.jsonl')+read(D/'results/obf_eval/obf2g-s25-e25-french/rows.jsonl');post=read(D/'results/obf_eval/rwsft1-ep2-e25/rows.jsonl');raw['ds_pre']=pre;raw['ds_post']=post
for t in sorted({x['type'] for x in post} - {'nocot'}):
 a={x['qid']:x for x in pre if x['type']==t};b={x['qid']:x for x in post if x['type']==t};assert a.keys()==b.keys();n=len(a);out={'n':n}
 for label,rr in [('before',a),('after',b)]:
  for r in rr.values():
   if t!='baseline':assert bool(r['passed'])==bool(r['fmt_ok'] and r['strict'] and r['gate'] is True and r['meta'] is True)
  k=sum(bool(r['passed']) for r in rr.values());out[label]={'pass':k,'ci95':wilson(k,n),'correct':sum(bool(r['correct']) for r in rr.values()),'joint':sum(bool(r['correct'] and r['passed']) for r in rr.values()),'truncated':sum(bool(r['truncated']) for r in rr.values()),'median_words':statistics.median(len(r['reasoning'].split()) for r in rr.values()),'pass_without_known_overlap':sum(bool(r['passed']) for q,r in rr.items() if q!='64613358-cafd-4156-8870-495632457c72')}
 diff=np.array([int(bool(b[q]['passed']))-int(bool(a[q]['passed'])) for q in sorted(a)]);idx=rng.integers(n,size=(10000,n));out['delta_ci95']=np.quantile(diff[idx].mean(1),[.025,.975]).tolist();data['deepseek'][t]=out
(O/'other_models_data.json').write_text(json.dumps(data,indent=2));(O/'other_models_sources.json').write_text(json.dumps(hashes,indent=2))
# Random raw review, recorded selections, ten positives and negatives for each new model stage.
r=random.Random(20261030);samples=[]
for run in runs[:6]:
 rr={x['key']:x for x in raw[run]}
 for passed in [0,1]:
  candidates=[x for x in raw[run+'_graded'] if x['strict']==passed]
  for x in r.sample(candidates,min(10,len(candidates))):samples.append({'run':run,**x,'prompt':rr[x['key']]['user_prompt'],'reasoning':rr[x['key']]['reasoning'],'response':rr[x['key']]['response']})
Path('work/other_models_raw_review.json').write_text(json.dumps(samples,indent=2))
samples=[]
for stage,rr in [('before',pre),('after',post)]:
 for passed in [False,True]:
  cand=[x for x in rr if x['type'] in ['sad','cover_story','happy','french'] and bool(x['passed'])==passed]
  for x in r.sample(cand,min(10,len(cand))):samples.append({'stage':stage,**x})
Path('work/deepseek_raw_review.json').write_text(json.dumps(samples,indent=2))
for run,x in data['astra'].items():print(run,{k:v for k,v in x.items() if k not in ['sampling']})
for t,x in data['deepseek'].items():print(t,x)
print('DS review word counts:',[(x['stage'],x['type'],x['passed'],len(x['reasoning'].split())) for x in samples])
