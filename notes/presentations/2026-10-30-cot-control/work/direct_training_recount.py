import ast, collections, hashlib, json, random, re, sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/john/code/nocotrl-b-pilot/cot-control'); INK=Path('/home/john/code/tlg-inksm/cot-control'); OUT=Path('outputs')
sys.path.insert(0,str(ROOT/'grading'))
from cotswitch import graders
hashes={}
def read(p):
 b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b) if p.suffix=='.json' else [json.loads(l) for l in b.splitlines() if l.strip()]
ga={}
for p in [ROOT/'results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl', ROOT/'results/outctl/new_items.json']:
 for r in read(p):ga[r['key']]=ast.literal_eval(r['grader_args']) if isinstance(r['grader_args'],str) else r['grader_args']
allrows={};items={};clusters={};result={};reviews=[];rng=random.Random(20261030)
for model in ['base','grpo3','lang1-s40','len2']:
 rs=[]
 for part in ['more','big']:
  for r in read(ROOT/f'results/outctl/{part}/cot/{model}/graded.json'):
   if r['constraint']=='ignore_question':continue
   k=r['key'];args=ga.get(k) or ga.get(k.rsplit('|',1)[0]);assert args is not None
   fresh=int(graders.grade('astra',r['constraint'],r['body'],args)['strict'] or 0);assert fresh==r['strict'],(model,k)
   r['uid']=part+'|'+k
   q=r['user'].split('\n\nA)')[0];cluster=re.sub(r'[^a-z0-9]','',q.lower())
   clusters[r['uid']]=cluster;rs.append(r)
 allrows[model]=rs;grouped=collections.defaultdict(list)
 for r in rs:grouped[r['uid']].append(r)
 items[model]={k:float(np.mean([r['strict'] for r in g])) for k,g in grouped.items()}
 result[model]={'samples':len(rs),'items':len(grouped),'mean_strict':float(np.mean(list(items[model].values()))),'accuracy':float(np.mean([r['correct'] for r in rs]))}
 for positive in [False,True]:
  pool=[r for r in rs if bool(r['strict'])==positive]
  for r in rng.sample(pool,min(10,len(pool))):reviews.append(dict(model=model,key=r['key'],sample=r['sample_idx'],constraint=r['constraint'],strict=r['strict'],body=r['body'][:350],tail=r['body'][-130:],response=r['resp'][:100]))
keys=sorted(items['base']);assert all(set(d)==set(keys) for d in items.values())
cids=sorted(set(clusters[k] for k in keys));index={c:i for i,c in enumerate(cids)};counts=np.zeros(len(cids));sums={m:np.zeros(len(cids)) for m in items}
for k in keys:
 i=index[clusters[k]];counts[i]+=1
 for m in items:sums[m][i]+=items[m][k]
ix=np.random.default_rng(20261030).integers(len(cids),size=(5000,len(cids)));denom=counts[ix].sum(axis=1);boots={m:s[ix].sum(axis=1)/denom for m,s in sums.items()}
for m in result:
 result[m]['ci95']=np.percentile(boots[m],[2.5,97.5]).tolist();result[m]['delta_vs_base']=result[m]['mean_strict']-result['base']['mean_strict'];result[m]['delta_ci95']=np.percentile(boots[m]-boots['base'],[2.5,97.5]).tolist()
backup={}
for n in ['inksmtk-up-e06','inksft1-ep2-req-e06','qwen38-run2','opsd1-step30-t1','opsd-grpo1-step30-t1']:
 rows=read(next((INK/'results/tinker-runs'/n/'generations').glob('*/astra.jsonl')));rows=[r for r in rows if r['constraint']!='ignore_question']
 for r in rows:
  r['s']=int(graders.grade('astra',r['constraint'],r['reasoning'],r['grader_args'])['strict'] or 0)
  a=re.search(r'ANSWER\s*:\s*([A-Z])\b',r['response'] or '',re.I)
  r['v']=bool(r['reasoning'].strip()) and bool(r['think_closed']) and r['finish_reason']!='length' and a is not None
 backup[n]={'n':len(rows),'strict':sum(r['s'] for r in rows),'strict_valid':sum(r['s'] and r['v'] for r in rows),'joint':sum(r['s'] and r['v'] and r['correct'] for r in rows),'correct':sum(r['correct'] for r in rows)}
 for positive in [False,True]:
  pool=[r for r in rows if bool(r['s'])==positive]
  for r in rng.sample(pool,min(10,len(pool))):reviews.append(dict(model=n,key=r['key'],constraint=r['constraint'],strict=r['s'],body=r['reasoning'][:350],tail=r['reasoning'][-130:],response=r['response'][:100]))
meta={'main':result,'backup':backup,'question_clusters':len(cids),'interval':'5000 paired question-cluster percentile bootstrap draws, seed 20261030; equal weight to each of 480 question-rule items after averaging samples; intervals reflect evaluation-question uncertainty, not training-seed variation','metric':'Strict deterministic CoT rule compliance; meta discussion allowed; no correctness or completion gate; semantic ignore_question excluded','hashes':hashes}
(OUT/'direct_training_data.json').write_text(json.dumps(meta,indent=2));Path('work/direct_training_samples.json').write_text(json.dumps(reviews,indent=2))
print(json.dumps({k:v for k,v in meta.items() if k!='hashes'},indent=2))
