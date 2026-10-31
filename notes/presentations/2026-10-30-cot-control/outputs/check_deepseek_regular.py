import json,ast,re,sys,collections,hashlib
from pathlib import Path
import numpy as np
C=Path('/home/john/code/tlg-arena-rl/cot-control');O=C/'results/outctl';sys.path.insert(0,str(C/'grading'))
from cotswitch import graders
items={x['key']:x for x in json.loads((O/'all_items_rs.json').read_text())};out={};hashes={};review=[]
rng=np.random.default_rng(20261030)
for side in ['cot','cot_low']:
 for model in ['ds-base-tk','ds-obf2g','ds-rwsft1']:
  p=O/f'xfull/{side}/{model}/gens.jsonl';b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();raw=[json.loads(l) for l in b.splitlines()];rr=[]
  stored={ (x['key'],x['sample_idx']):x['strict'] for x in json.loads((p.parent/'strict_graded.json').read_text())};mm=[]
  for r in raw:
   if r['constraint']=='ignore_question':continue
   a=items[r['key']]['grader_args'];a=ast.literal_eval(a) if isinstance(a,str) else a;a={k:v for k,v in a.items() if k not in ['prompt','user_prompt']}
   for k in ['synonyms','keywords']:
    if isinstance(a.get(k),str):a[k]=ast.literal_eval(a[k])
   s=int(bool(r['out'].strip()) and graders._grade_default(r['constraint'],r['out'].strip(),a)['strict']==1)
   if s!=stored[(r['key'],r['sample_idx'])]:mm.append(r['key'])
   r['strict']=s;rr.append(r)
  assert not mm,mm
  for subset in ['all','astra']:
   rs=[r for r in rr if subset=='all' or r['key'].startswith('astra|')];group=collections.defaultdict(list)
   for r in rs:group[r['key']].append(r)
   ks=sorted(group);vs=np.array([np.mean([r['strict'] for r in group[k]]) for k in ks]);q={k:re.sub(r'[^a-z0-9]','',group[k][0]['user'].split('\n\nA)')[0].lower()) for k in ks};qs=sorted(set(q.values()));ns=np.array([sum(q[k]==z for k in ks) for z in qs]);sums=np.array([sum(v for k,v in zip(ks,vs) if q[k]==z) for z in qs]);ix=rng.integers(len(qs),size=(10000,len(qs)));bs=sums[ix].sum(1)/ns[ix].sum(1)
   out[f'{side}/{model}/{subset}']={'items':len(ks),'samples':len(rs),'passes':sum(r['strict'] for r in rs),'rate':float(vs.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'clusters':len(qs),'bytype':{t:{'n':sum(r['constraint']==t for r in rs),'strict':sum(r['strict'] for r in rs if r['constraint']==t)} for t in sorted(set(r['constraint'] for r in rs))}}
  if side=='cot' and model!='ds-base-tk':
   for v in [0,1]:
    pool=[r for r in rr if r['strict']==v]
    for idx in rng.choice(len(pool),size=10,replace=False):
     r=pool[idx];review.append({'model':model,'key':r['key'],'strict':v,'reasoning':r['out'][:180],'end':r['out'][-80:]})
Path('outputs/deepseek_regular_control.json').write_text(json.dumps({'results':out,'hashes':hashes,'ci':'95% question-cluster percentile bootstrap; 10000 draws; seed 20261030; sampling uncertainty only'},indent=2));Path('work/deepseek_regular_review.json').write_text(json.dumps(review,indent=2))
for k,x in out.items():print(k,{z:v for z,v in x.items() if z!='bytype'})
