import json,sys,collections,re,numpy as np
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl/b_pilot','/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
from cotswitch import graders as G
from astra_continuous import score
from analyze import independent_output
P='/home/john/code/results/nocotrl-b-pilot/';F=P+'followup_noise_elicit/'
def load(*fs):
  d=collections.defaultdict(list)
  for f in fs:
    for l in open(f):
      r=json.loads(l);s=G.grade('astra',r['constraint'],r['cot'],r['grader_args'])['strict']
      if s is not None:d[r['key']].append((s,score(dict(constraint=r['constraint'],reasoning=r['cot'],grader_args=r['grader_args']),s),r['constraint']))
  return d
base=load(P+'astra_full_base.jsonl',F+'noise_base.jsonl');K=sorted(base);rng=np.random.default_rng(271029)
runs={'old s24':P+'astra_full_pilot.jsonl','old s36':P+'astra_full_s36.jsonl','old s60':P+'astra_full_s60.jsonl','add s12':P+'additive/astra_full_s12.jsonl','add s24':P+'additive/astra_full_s24.jsonl','add s36 (4 draws)':[P+'additive/astra_full_s36.jsonl',F+'noise_s36.jsonl']}
trained=('lowercase_thinking','multiple_word_suppression')
print('Astra partial compliance, delta vs base (base = mean of 4 draws per question), paired bootstrap over 96 questions')
for n,f in runs.items():
  d=load(*(f if isinstance(f,list) else [f]))
  for sub,keys in (('all',K),('trained fams',[k for k in K if base[k][0][2] in trained])):
    x=np.array([np.mean([t[1] for t in d[k]])-np.mean([t[1] for t in base[k]]) for k in keys]);bs=rng.choice(x,(10000,len(x))).mean(1)
    print(f'  {n:18} {sub:12} {x.mean():+.3f} [{np.quantile(bs,.025):+.3f},{np.quantile(bs,.975):+.3f}]')
print('Dev output compliance (held-out phrasing, constrained tasks), strict_robust / joint_robust')
tasks={json.loads(l)['id']:json.loads(l) for l in open(P+'dev.jsonl')}
for n,f in [('base',P+'dev_base.jsonl'),('old s24',P+'dev_pilot.jsonl'),('old s36',P+'dev_s36.jsonl'),('old s60',P+'dev_s60.jsonl'),('add s12',P+'additive/dev_s12.jsonl'),('add s24',P+'additive/dev_s24.jsonl'),('add s36',P+'additive/dev_s36.jsonl')]:
  R=[json.loads(l) for l in open(f)];S=[independent_output(tasks[r['task_id']],r) for r in R if tasks[r['task_id']]['rule']!='none']
  print(f'  {n:8} strict {sum(s["strict_robust"] for s in S)}/{len(S)}  joint {sum(s["joint_robust"] for s in S)}/{len(S)}')
