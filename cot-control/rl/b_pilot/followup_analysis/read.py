import json,random,sys
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/grading']
from cotswitch import graders as G
P='/home/john/code/results/nocotrl-b-pilot/'
B={json.loads(l)['key']:json.loads(l) for l in open(P+'astra_full_base.jsonl')}
S={json.loads(l)['key']:json.loads(l) for l in open(P+'additive/astra_full_s36.jsonl')}
rng=random.Random(271028)
for c in sorted({r['constraint'] for r in S.values()}-{'ignore_question'}):
  k=rng.choice([k for k in S if S[k]['constraint']==c])
  print(f'##### {c}  {k}')
  for lab,D in (('BASE',B),('S36',S)):
    r=D[k];st=G.grade('astra',c,r['cot'],r['grader_args'])['strict']
    t=r['cot'].replace('\n',' / ');print(f'  {lab} strict={st} words={len(r["cot"].split())}: {t[:260]!r} ... {t[-120:]!r}')
