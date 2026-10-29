import json,sys,collections,numpy as np
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
from cotswitch import graders as G
from astra_continuous import score
P='/home/john/code/results/nocotrl-b-pilot/';F=P+'followup_noise_elicit/'
def load(*fs):
  d=collections.defaultdict(list)
  for f in fs:
    for l in open(f):
      r=json.loads(l);s=G.grade('astra',r['constraint'],r['cot'],r['grader_args'])['strict']
      if s is not None:d[r['key']].append((s,score(dict(constraint=r['constraint'],reasoning=r['cot'],grader_args=r['grader_args']),s)))
  return d
b=load(P+'astra_full_base.jsonl',F+'noise_base.jsonl');s=load(P+'additive/astra_full_s36.jsonl',F+'noise_s36.jsonl');fs=load(F+'fewshot.jsonl')
K=sorted(b);rng=np.random.default_rng(271028)
def cmp(name,x,y,i):
  d=np.array([np.mean([t[i] for t in y[k]])-np.mean([t[i] for t in x[k]]) for k in K]);bs=rng.choice(d,(10000,len(d))).mean(1)
  print(f'{name}: delta {d.mean():+.3f} 95% CI [{np.quantile(bs,.025):+.3f}, {np.quantile(bs,.975):+.3f}] (n={len(d)} questions)')
cmp('s36 - base, strict rate',b,s,0);cmp('s36 - base, partial',b,s,1);cmp('fewshot - base, strict rate',b,fs,0);cmp('fewshot - base, partial',b,fs,1)
# prefill: where does lowercase/uppercase continuation break?
for l in open(F+'prefill.jsonl'):
  r=json.loads(l)
  if r['constraint']=='lowercase_thinking' and r['sample']==0:
    c=r['cot'];print('PREFILL lc:',repr(c[:160]))
