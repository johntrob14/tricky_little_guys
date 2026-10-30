"""Paired comparison of two models on the same elicit.py variants (test split): strict, partial (astra_continuous on the stored CoT text,
or on the answer for outnocot), median CoT words, truncation, accuracy; per-type strict for one variant. Usage: elicit_pair.py BASE NEW V1,V2,..."""
import json,sys,collections,re,numpy as np
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
from astra_continuous import score
E='/home/john/code/tlg-inksm/cot-control/results/outctl/elicit/runs/'
GA={json.loads(l)['key']:json.loads(l)['grader_args'] for l in open('/home/john/code/nocotrl-b-pilot/cot-control/results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl')}
def load(m,v):
  d=collections.defaultdict(list)
  for r in json.load(open(E+f'{m}/{v}_test/graded.json')):
    st=float(str(r.get('strict')) in('1','True'));txt=(r.get('out') or '')
    pa=st if r['constraint']=='ignore_question' else score(dict(constraint=r['constraint'],reasoning=txt,grader_args=GA[r['key']]),st)
    d[r['key']].append(dict(s=st,p=pa,w=len(txt.split()),t=r.get('stop')=='length',c=str(r.get('correct')) in('1','True'),k=r['constraint']))
  return d
b_,n_,vs=sys.argv[1],sys.argv[2],sys.argv[3].split(',');rng=np.random.default_rng(13)
print(f"{'variant':11}{'strict '+b_:>16}{n_:>12}{'d strict [CI]':>22}{'partial '+b_:>18}{n_:>12}{'d partial [CI]':>22}{'medW':>12}{'trunc':>10}{'acc':>12}")
for v in vs:
  b,n=load(b_,v),load(n_,v);K=sorted(set(b)&set(n));row=f"{v:11}"
  for f in ('s','p'):
    bm=np.mean([x[f] for k in K for x in b[k]]);nm=np.mean([x[f] for k in K for x in n[k]])
    x=np.array([np.mean([t[f] for t in n[k]])-np.mean([t[f] for t in b[k]]) for k in K]);bs=rng.choice(x,(10000,len(x))).mean(1)
    row+=f"{100*bm:15.1f}%{100*nm:11.1f}%  {100*x.mean():+5.1f} [{100*np.quantile(bs,.025):+5.1f},{100*np.quantile(bs,.975):+5.1f}]"
  med=lambda d:int(np.median([x['w'] for k in K for x in d[k]]))
  row+=f"  {med(b)}/{med(n)}  {sum(x['t'] for k in K for x in b[k])}/{sum(x['t'] for k in K for x in n[k])}  {100*np.mean([x['c'] for k in K for x in b[k]]):.0f}/{100*np.mean([x['c'] for k in K for x in n[k]]):.0f}%"
  print(row)
for v in vs[:1]+(['baselow'] if 'baselow' in vs else []):
  for m in (b_,n_):
    by=collections.defaultdict(list)
    for k,xs in load(m,v).items():
      for x in xs:by[x['k']].append(x['s'])
    print(f"  {v} {m}: "+' '.join(f"{c[:6]}:{int(sum(s))}/{len(s)}" for c,s in sorted(by.items())))
