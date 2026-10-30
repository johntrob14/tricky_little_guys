"""Paired (per-item) differences on Astra's 96 rule-graded items: variant arms and s100 vs their s90 start, 4 draws each."""
import json,sys,collections,numpy as np
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
from cotswitch import graders as G
from astra_continuous import score
P='/home/john/code/results/nocotrl-b-pilot/'
def load(*fs):
  d=collections.defaultdict(list)
  for f in fs:
    for l in open(f):
      r=json.loads(l);s=G.grade('astra',r['constraint'],r['cot'],r['grader_args'])['strict']
      if s is not None:d[r['key']].append((s,score(dict(constraint=r['constraint'],reasoning=r['cot'],grader_args=r['grader_args']),s),r['correct']))
  return d
s90=load(P+'additive/astra_full_s90.jsonl',P+'followup_noise_elicit/draws_add_s90/noise_add_s90.jsonl');K=sorted(s90);rng=np.random.default_rng(4)
arms={'s100 (same data)':load(P+'additive/astra_full_s100.jsonl',P+'followup_noise_elicit/draws_add_s100/noise_add_s100.jsonl'),
      'fresh s102':load(P+'variants/fresh/astra_full_s102.jsonl',P+'variants/fresh/draws/noise_fresh_s102.jsonl'),
      'stacked s102':load(P+'variants/stacked/astra_full_s102.jsonl',P+'variants/stacked/draws/noise_stacked_s102.jsonl'),
      'fresh s114':load(P+'variants/fresh/astra_full_s114.jsonl',P+'variants/fresh/draws2/noise_fresh_s114.jsonl'),
      'stacked s114':load(P+'variants/stacked/astra_full_s114.jsonl',P+'variants/stacked/draws2/noise_stacked_s114.jsonl')}
for n,d in arms.items():
  out=[]
  for i,m in ((0,'strict'),(1,'partial'),(2,'correct')):
    x=np.array([np.mean([t[i] for t in d[k]])-np.mean([t[i] for t in s90[k]]) for k in K]);bs=rng.choice(x,(10000,len(x))).mean(1)
    out.append(f'{m} {x.mean()*100:+.1f} [{np.quantile(bs,.025)*100:+.1f},{np.quantile(bs,.975)*100:+.1f}]')
  print(f'{n:18} minus s90: '+'  '.join(out))
