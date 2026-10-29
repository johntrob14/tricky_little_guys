import json,sys,collections,re,numpy as np
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl/b_pilot','/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
from cotswitch import graders as G
from astra_continuous import score
from analyze import independent_output
P='/home/john/code/results/nocotrl-b-pilot/';F=P+'followup_noise_elicit/'
def first(t):return re.split(r'(?<=[.!?:])\s+|\n+',t.strip())[0]
def load(*fs):
  d=collections.defaultdict(list)
  for f in fs:
    for l in open(f):
      r=json.loads(l);s=G.grade('astra',r['constraint'],r['cot'],r['grader_args'])['strict']
      if s is None:continue
      on=G.grade('astra',r['constraint'],first(r['cot']),r['grader_args'])['strict'] if r['constraint']=='lowercase_thinking' else None
      d[r['key']].append(dict(s=s,p=score(dict(constraint=r['constraint'],reasoning=r['cot'],grader_args=r['grader_args']),s),c=r['constraint'],on=on,ok=r['correct']))
  return d
base=load(P+'astra_full_base.jsonl',F+'noise_base.jsonl');K=sorted(base);rng=np.random.default_rng(271030)
runs=[('old s36 (1)',[P+'astra_full_s36.jsonl']),('old s60 (4)',[P+'astra_full_s60.jsonl',F+'draws_old_s60/noise_old_s60.jsonl']),
      ('add s36 (4)',[P+'additive/astra_full_s36.jsonl',F+'noise_s36.jsonl']),('add s48 (4)',[P+'additive/astra_full_s48.jsonl',F+'draws_add_s48/noise_add_s48.jsonl']),('add s60 (4)',[P+'additive/astra_full_s60.jsonl',F+'draws_add_s60/noise_add_s60.jsonl'])]
D={'base (4)':base,**{n:load(*f) for n,f in runs}}
def ci(x):bs=rng.choice(x,(10000,len(x))).mean(1);return f'{x.mean():+.3f} [{np.quantile(bs,.025):+.3f},{np.quantile(bs,.975):+.3f}]'
trained=('lowercase_thinking','multiple_word_suppression')
print(f"{'run':12} {'strict/draws':>14} {'partial':>7} {'d partial vs base (all)':>28} {'d partial trained fams':>28} {'lc onset':>9} {'correct':>8}")
for n,d in D.items():
  allr=[t for k in K for t in d[k]];nd=len(allr)/96
  x=np.array([np.mean([t['p'] for t in d[k]])-np.mean([t['p'] for t in base[k]]) for k in K])
  xt=np.array([np.mean([t['p'] for t in d[k]])-np.mean([t['p'] for t in base[k]]) for k in K if base[k][0]['c'] in trained])
  on=[t['on'] for t in allr if t['on'] is not None]
  print(f"{n:12} {sum(t['s'] for t in allr):>5}/{len(allr):<4}({nd:.0f}) {np.mean([t['p'] for t in allr]):7.3f} {ci(x):>28} {ci(xt):>28} {sum(on):>3}/{len(on):<3}  {np.mean([t['ok'] for t in allr]):.3f}")
x=np.array([np.mean([t['p'] for t in D['add s60 (4)'][k]])-np.mean([t['p'] for t in D['old s60 (4)'][k]]) for k in K]);print('add s60 - old s60 partial:',ci(x))
print('Dev output compliance (53 constrained, held-out phrasing): strict_robust / joint_robust')
tasks={json.loads(l)['id']:json.loads(l) for l in open(P+'dev.jsonl')}
for n,f in [('base',P+'dev_base.jsonl'),('old s36',P+'dev_s36.jsonl'),('old s60',P+'dev_s60.jsonl'),('add s36',P+'additive/dev_s36.jsonl'),('add s48',P+'additive/dev_s48.jsonl'),('add s60',P+'additive/dev_s60.jsonl')]:
  R=[json.loads(l) for l in open(f)];S=[independent_output(tasks[r['task_id']],r) for r in R if tasks[r['task_id']]['rule']!='none']
  print(f'  {n:8} strict {sum(s["strict_robust"] for s in S)}/{len(S)}  joint {sum(s["joint_robust"] for s in S)}/{len(S)}')
