import ast,collections,hashlib,json,re,random,sys
from pathlib import Path
import numpy as np
R=Path('/home/john/code/nocotrl-b-pilot/cot-control');O=Path('outputs');hashes={};reviews=[];rnd=random.Random(20261030)
def read(p):
 b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b) if p.suffix=='.json' else [json.loads(x) for x in b.splitlines() if x]
sys.path.insert(0,str(R/'grading'));from cotswitch import graders
tr=ast.parse((R/'rl/controls.py').read_text());ns={'re':re};nodes=[n for n in tr.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['LANG_STOP','LANG_SENT_RE'] for t in n.targets) or isinstance(n,ast.FunctionDef) and n.name=='lang_profile'];exec(compile(ast.Module(body=nodes,type_ignores=[]),'classifier','exec'),ns);profile=ns['lang_profile']
ga={}
for p in [R/'results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl',R/'results/outctl/new_items.json']:
 for r in read(p):ga[r['key']]=ast.literal_eval(r['grader_args']) if isinstance(r['grader_args'],str) else r['grader_args']
models=['base','lang1-s20','lang1-s40','lang2-s18','lang-ans1-s20','langsft1-s8','langsft1-s20','langsft1-en-s20','langsft1-back-s20','langsft1-ds-s20','langsft1-noise-s20','langsft1-end-s20'];items={};cs={};counts={}
for m in models:
 by=collections.defaultdict(list);rr=[]
 for part in ['more','big']:
  p=R/f'results/outctl/{part}/cot/{m}/graded.json'
  if not p.exists():continue
  for r in read(p):
   if r['constraint']=='ignore_question':continue
   k=r['key'];args=ga.get(k) or ga.get(k.rsplit('|',1)[0]);fresh=int(graders.grade('astra',r['constraint'],r['body'],args)['strict'] or 0);assert fresh==r['strict'];uid=part+'|'+k
   by[uid].append(fresh);cs[uid]=re.sub('[^a-z0-9]','',r['user'].split('\n\nA)')[0].lower());rr.append(r)
 items[m]={k:np.mean(v) for k,v in by.items()};counts[m]=len(rr)
 for ok in [0,1]:
  pool=[r for r in rr if r['strict']==ok]
  for r in rnd.sample(pool,min(10,len(pool))):reviews.append({'category':'cot','model':m,'strict':ok,'rule':r['constraint'],'text':r['body'][:240],'end':r['body'][-80:]})
result={};rng=np.random.default_rng(20261030)
for m,it in items.items():
 keys=sorted(it);groups=sorted({cs[k] for k in keys});n=np.array([sum(cs[k]==g for k in keys) for g in groups]);s=np.array([sum(it[k] for k in keys if cs[k]==g) for g in groups]);b=np.array([sum(items['base'][k] for k in keys if cs[k]==g) for g in groups]);ix=rng.integers(len(groups),size=(5000,len(groups)));diff=((s-b)[ix].sum(1)/n[ix].sum(1));result[m]={'items':len(it),'samples':counts[m],'rate':float(np.mean(list(it.values()))),'matched_base':float(np.mean([items['base'][k] for k in keys])),'gain_ci95':np.quantile(diff,[.025,.975]).tolist()}
lang={}
for r in read(R/'results/outctl/qlang/answer_lang.jsonl'):
 p=profile(r['reasoning']);v=bool(len(p)>=3 and sum(x==r['lang'] for x in p)/len(p)>=.95);assert v==r['r_strict'];lang.setdefault(r['model']+'/'+r['lang'],[]).append(r)
lout={}
for k,rs in lang.items():
 lout[k]={'n':len(rs),'strict':sum(r['r_strict'] for r in rs),'truncations':sum(r['stop']=='length' for r in rs)}
 if k.endswith('/Italian') and 'step20' in k:
  for v in [0,1]:
   pool=[r for r in rs if r['r_strict']==v]
   for r in rnd.sample(pool,min(10,len(pool))):reviews.append({'category':'language','model':k,'strict':v,'text':r['reasoning'][:300]})
train={}
for m in ['lang1','lang2','lang-ans1','fr-hot','fr-shaped','fr-mix','fr-from-lang1']:
 rs=read(R/f'results/rl/{m}/rollouts.jsonl');steps=sorted({r['step'] for r in rs});train[m]={'steps':steps,'n':len(rs),'strict_total':sum(r['strict'] for r in rs),'first':{},'last':{}}
 for tag,step in [('first',steps[0]),('last',steps[-1])]:
  x=[r for r in rs if r['step']==step];train[m][tag]={'step':step,'n':len(x),'strict':sum(r['strict'] for r in x),'correct':sum(r['correct'] for r in x)}
 if m.startswith('fr-'):
  for ok in [0,1]:
   pool=[r for r in rs if bool(r['strict'])==ok]
   for r in rnd.sample(pool,min(10,len(pool))):reviews.append({'category':'training','model':m,'strict':ok,'text':r['reasoning'][:220]})
configs={m:read(R/f'results/rl/{m}/config.json') for m in ['lang1','lang2','lang-ans1','fr-hot','fr-shaped','fr-mix','fr-from-lang1','langsft1','langsft1-en','langsft1-back','langsft1-ds','langsft1-noise','langsft1-end']}
(O/'language_variants_audit.json').write_text(json.dumps({'cot':result,'language_noformat':lout,'training':train,'configs':configs,'hashes':hashes,'seed':20261030},indent=2));Path('work/language_variants_review.json').write_text(json.dumps(reviews,indent=2));print(json.dumps({'cot':result,'language_noformat':lout,'training':train},indent=2))
