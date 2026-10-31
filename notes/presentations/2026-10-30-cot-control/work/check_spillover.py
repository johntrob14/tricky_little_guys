import ast,json,sys,importlib.util,collections,statistics,random
from pathlib import Path
import numpy as np
root=Path('/home/john/code/nocotrl-b-pilot/cot-control');out=Path('outputs')
sys.path.insert(0,str(root/'grading'))
spec=importlib.util.spec_from_file_location('ac',root/'rl/astra_continuous.py');ac=importlib.util.module_from_spec(spec);spec.loader.exec_module(ac)
ga={}
for path in [root/'results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl',root/'results/outctl/new_items.json']:
 rows=[json.loads(l) for l in path.open()] if path.suffix=='.jsonl' else json.loads(path.read_text())
 for r in rows:ga[r['key']]=ast.literal_eval(r['grader_args']) if isinstance(r['grader_args'],str) else r['grader_args']
summary={}; allrows={}
for model in ['base','outrl1-s53']:
 rs=[]
 for part in ['more','big']:
  rows=json.loads((root/f'results/outctl/{part}/cot/{model}/graded.json').read_text())
  for r in rows:
   if r['constraint']=='ignore_question':continue
   k=r['key'];args=ga.get(k) or ga.get(k.rsplit('|',1)[0]) or {}
   r['partial_new']=ac.score(dict(constraint=r['constraint'],reasoning=r['body'],grader_args=args),r['strict']);r['uid']=(part,k)
   rs.append(r)
 allrows[model]=rs
 grouped=collections.defaultdict(list)
 for r in rs:grouped[r['uid']].append(r)
 summary[model]={k:{metric:statistics.mean(float(r[metric]) for r in rr) for metric in ['strict','partial_new']} for k,rr in grouped.items()}
keys=sorted(summary['base']);assert set(keys)==set(summary['outrl1-s53'])
rng=np.random.default_rng(20261030);ix=rng.integers(0,len(keys),(5000,len(keys)))
result={'scope':'480 deterministic-rule items; semantic ignore_question excluded; equal weight per item','samples':{m:len(rs) for m,rs in allrows.items()},'metrics':{}}
for metric in ['strict','partial_new']:
 a=np.array([summary['base'][k][metric] for k in keys]);b=np.array([summary['outrl1-s53'][k][metric] for k in keys]);delta=b-a
 result['metrics'][metric]={'before':float(a.mean()),'after':float(b.mean()),'delta':float(delta.mean()),'ci95':np.percentile(delta[ix].mean(axis=1),[2.5,97.5]).tolist()}
result['limitations']=['Single training run','Frozen partial-score implementation reused; fresh item aggregation','No causal separation of shared-parameter transfer and feedback spillover','Raw review uses seeded excerpts, no full factual-accuracy audit','No fresh inference or training']
(out/'spillover_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
rng2=random.Random(20261030);review=[]
for model,rs in allrows.items():
 for positive in [False,True]:
  pool=[r for r in rs if (r['partial_new']>0.5)==positive]
  for r in rng2.sample(pool,10):
   review.append({'model':model,'key':r['key'],'sample_idx':r['sample_idx'],'constraint':r['constraint'],'partial':r['partial_new'],'strict':r['strict'],'excerpt':r['body'][:450]})
Path('work/spillover_samples.json').write_text(json.dumps(review,indent=2))
