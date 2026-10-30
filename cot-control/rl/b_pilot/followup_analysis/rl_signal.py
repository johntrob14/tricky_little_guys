import json,collections,numpy as np
P='/home/john/code/results/nocotrl-b-pilot/'
def rows(fs,maxstep):
  out=[]
  for f in fs:
    for l in open(P+f):
      r=json.loads(l)
      if r['step']<=maxstep and not r['grade'].get('has_generated_analysis'):out.append(r)
  return out
old=rows(['train_rollouts.jsonl','train_rollouts_resume.jsonl','train_rollouts_continuation36.jsonl','train_rollouts_continuation60.jsonl'],60)
# old base log has steps 1-24 pre-restart possibly duplicated; dedupe on (step,task,rep)
seen=set();o2=[]
for r in old:
  k=(r['step'],r['task_id'],r['rep'])
  if k not in seen:seen.add(k);o2.append(r)
old=o2
add=rows(['additive/train_rollouts.jsonl','additive/train_rollouts_to24.jsonl','additive/train_rollouts_to36.jsonl','additive/train_rollouts_to48.jsonl'],39)
mult=lambda g:float(g['quality'])*(1+g['partial']+.5*g['strict'])
addr=lambda g:1.5*float(g['quality'])+g['partial']+.5*g['strict']
def stats(R,rew,lo,hi):
  G=collections.defaultdict(list)
  for r in R:
    if lo<=r['step']<=hi and r['rule']!='none':G[(r['step'],r['task_id'])].append(r['grade'])
  cs=[];cq=[];n=0;nz=0;wrongfmt=[]
  for g in G.values():
    R_=np.array([rew(x) for x in g]);A=R_-R_.mean();S=np.array([x['partial']+.5*x['strict'] for x in g]);Q=np.array([float(x['quality']) for x in g])
    cs.append((A*(S-S.mean())).mean());cq.append((A*(Q-Q.mean())).mean());n+=1;nz+=np.abs(A).max()>1e-9
  return np.mean(cs),np.mean(cq),nz/n
print('constrained tasks only; signal = mean within-group cov(advantage, X)')
for name,R in (('old run rollouts',old),('additive rollouts',add)):
  for lo,hi in ((1,12),(13,24),(25,39)):
    for rn,rw in (('mult',mult),('add',addr)):
      fs,tq,frac=stats(R,rw,lo,hi)
      print(f'{name:18} steps {lo:2}-{hi:2} reward={rn:4}: format signal {fs:.4f}  task signal {tq:.4f}  groups w/ nonzero adv {frac:.2f}')
for name,R in (('old',old),('add',add)):
  for lo,hi in ((1,12),(25,39)):
    g=[r['grade'] for r in R if lo<=r['step']<=hi and r['rule']!='none']
    print(f'{name} steps {lo}-{hi}: quality {np.mean([x["quality"] for x in g]):.3f} strict {np.mean([x["strict"] for x in g]):.3f} partial {np.mean([x["partial"] for x in g]):.3f} strict|wrong {np.mean([x["strict"] for x in g if not x["quality"]]):.3f}')
