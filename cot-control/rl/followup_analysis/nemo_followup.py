"""Nemotron-3 Super base vs nemo1 s80 follow-ups: (1) shotstack at default effort pooled over the 2-sample and 4-sample runs (6 per item),
paired over items, all samples and untruncated only, by type; (2) low-effort elicitation variants (test split); (3) low-effort Astra 108
via hf_astra_score.py."""
import json,sys,collections,subprocess,numpy as np
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
from astra_continuous import score
E='/home/john/code/tlg-inksm/cot-control/results/outctl/elicit/runs/'
GA={json.loads(l)['key']:json.loads(l)['grader_args'] for l in open('/home/john/code/nocotrl-b-pilot/cot-control/results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl')}
rng=np.random.default_rng(15)
def load(dirs):
  d=collections.defaultdict(list)
  for p in dirs:
    for r in json.load(open(E+p+'/graded.json')):
      st=float(str(r.get('strict')) in('1','True'));t=r.get('out') or ''
      pa=st if r['constraint']=='ignore_question' else score(dict(constraint=r['constraint'],reasoning=t,grader_args=GA[r['key']]),st)
      d[r['key']].append(dict(s=st,p=pa,tr=r.get('stop')=='length',w=len(t.split()),c=r['constraint']))
  return d
def compare(name,b,n,keep=lambda x:True):
  K=sorted(k for k in set(b)&set(n) if any(keep(x) for x in b[k]) and any(keep(x) for x in n[k]));out=f"{name:34}"
  for f in ('s','p'):
    bm=np.mean([x[f] for k in K for x in b[k] if keep(x)]);nm=np.mean([x[f] for k in K for x in n[k] if keep(x)])
    x=np.array([np.mean([t[f] for t in n[k] if keep(t)])-np.mean([t[f] for t in b[k] if keep(t)]) for k in K]);bs=rng.choice(x,(10000,len(x))).mean(1)
    out+=f"{'strict' if f=='s' else 'partial'} {100*bm:4.1f}->{100*nm:4.1f} ({100*x.mean():+.1f} [{100*np.quantile(bs,.025):+.1f},{100*np.quantile(bs,.975):+.1f}])  "
  w=lambda d:int(np.median([x['w'] for k in K for x in d[k] if keep(x)]))
  print(out+f"medW {w(b)}/{w(n)} n_items={len(K)}")
print('== (1) shotstack, default effort, 6 samples per item')
b=load(['nemosuper/shotstack_test','nemosuper_x4/shotstack_test']);n=load(['nemo1_s80/shotstack_test','nemo1_s80_x4/shotstack_test'])
compare('all samples',b,n);compare('untruncated only',b,n,lambda x:not x['tr'])
for nm,d in (('base',b),('s80',n)):
  by=collections.defaultdict(list)
  for k,xs in d.items():
    for x in xs:by[x['c']].append(x)
  print(f"  {nm:4} by type strict/partial: "+' '.join(f"{c[:6]}:{int(sum(x['s'] for x in v))}/{len(v)},p{100*np.mean([x['p'] for x in v]):.0f}" for c,v in sorted(by.items())))
print('== (2) low-effort elicitation, 2 samples per item')
for v in ('baselow','shotstacklow','thinklow'):
  try:compare(v,load([f'nemosuper/{v}_test']),load([f'nemo1_s80/{v}_test']))
  except FileNotFoundError as e:print(v,'missing',e)
print('== (3) Astra 108, low effort, 4 samples per item')
print(subprocess.run([sys.executable,'-I','/home/john/code/nocotrl-b-pilot/cot-control/rl/followup_analysis/hf_astra_score.py','base_low','nemo1_s80_low'],capture_output=True,text=True).stdout)
