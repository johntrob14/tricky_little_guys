import ast,collections,hashlib,json,random,re,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/john/code/nocotrl-b-pilot/cot-control');OUT=Path('outputs');sys.path.insert(0,str(ROOT/'grading'))
from cotswitch import graders
hashes={}
def read(p):
 b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b) if p.suffix=='.json' else [json.loads(l) for l in b.splitlines() if l.strip()]
ga={}
for p in [ROOT/'results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl',ROOT/'results/outctl/new_items.json']:
 for r in read(p):ga[r['key']]=ast.literal_eval(r['grader_args']) if isinstance(r['grader_args'],str) else r['grader_args']
items={};cluster={};result={};reviews=[];rng=random.Random(20261030)
for m in ['base','grpo3','lang1-s40','len2']:
 rs=[];groups=collections.defaultdict(list)
 for part in ['more','big']:
  path=ROOT/f'results/outctl/{part}/expl/{m}'
  raw={(r['key'],r['sample_idx']):r for r in read(path/'gens.jsonl')}
  rows=read(path/'graded.json');assert len(raw)==len(rows)
  for r in rows:
   if r['constraint']=='ignore_question':continue
   g=raw[r['key'],r['sample_idx']];body=re.sub(r'(<\|(return|end|call)\|>)+$','',g['out']).strip();assert body==r['body']
   a={k:v for k,v in ga[r['key']].items() if k not in ['prompt','user_prompt']}
   for k in ['synonyms','keywords']:
    if isinstance(a.get(k),str):a[k]=ast.literal_eval(a[k])
   score=int(bool(body) and graders._grade_default(r['constraint'],body,a)['strict'] or 0);assert score==r['strict'],(m,r['key'])
   uid=part+'|'+r['key'];groups[uid].append(r);rs.append(r)
   cluster[uid]=re.sub(r'[^a-z0-9]','',r['user'].split('\n\nA)')[0].lower())
 items[m]={u:float(np.mean([r['strict'] for r in g])) for u,g in groups.items()}
 result[m]={'samples':len(rs),'items':len(groups),'rate':float(np.mean(list(items[m].values()))),'mean_words':float(np.mean([r['words'] for r in rs])),'median_words':float(np.median([r['words'] for r in rs])),'strict_over5words':float(np.mean([np.mean([r['strict'] and r['words']>5 for r in g]) for g in groups.values()])), 'strict_complete':float(np.mean([np.mean([r['strict'] and r['stop']!='length' for r in g]) for g in groups.values()])),'bytype':{c:float(np.mean([np.mean([r['strict'] for r in g]) for g in groups.values() if g[0]['constraint']==c])) for c in sorted({r['constraint'] for r in rs})}}
 for v in [0,1]:
  pool=[r for r in rs if r['strict']==v]
  for r in rng.sample(pool,min(10,len(pool))):reviews.append(dict(model=m,key=r['key'],sample=r['sample_idx'],constraint=r['constraint'],strict=r['strict'],words=r['words'],stop=r['stop'],text=r['body'],prompt=r['user']))
keys=sorted(items['base']);assert all(set(it)==set(keys) for it in items.values())
cids=sorted(set(cluster[k] for k in keys));ci={c:i for i,c in enumerate(cids)};counts=np.zeros(len(cids));sums={m:np.zeros(len(cids)) for m in items}
for k in keys:
 i=ci[cluster[k]];counts[i]+=1
 for m in items:sums[m][i]+=items[m][k]
ix=np.random.default_rng(20261030).integers(len(cids),size=(5000,len(cids)));den=counts[ix].sum(axis=1);boots={m:v[ix].sum(axis=1)/den for m,v in sums.items()}
for m in result:
 result[m]['ci95']=np.percentile(boots[m],[2.5,97.5]).tolist();result[m]['delta']=result[m]['rate']-result['base']['rate'];result[m]['delta_ci95']=np.percentile(boots[m]-boots['base'],[2.5,97.5]).tolist()
meta={'models':result,'clusters':len(cids),'interval':'95% paired question-cluster percentile bootstrap, 5000 draws, seed 20261030; evaluation uncertainty only','metric':'Strict output rule compliance on the full visible answer and explanation; eight deterministic types; meta discussion allowed; no correctness or attempt gate','checkpoint_note':'All checkpoints match the aligned preceding slide: grpo3 s100, lang1 s40, len2 s60; no s30 explained-output evaluation located','hashes':hashes}
(OUT/'output_transfer_data.json').write_text(json.dumps(meta,indent=2));Path('work/output_transfer_samples.json').write_text(json.dumps(reviews,indent=2));print(json.dumps({k:v for k,v in meta.items() if k!='hashes'},indent=2))
