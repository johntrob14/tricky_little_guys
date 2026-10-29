"""Score hf_astra.py runs: strict/partial on the 96 rule-graded Astra items, paired vs base over items, CoT length, truncation, and the
length_control.py decomposition (functions exec'd verbatim) with base's own rows as the reference curve. Usage: hf_astra_score.py BASE NEW"""
import json,sys,os,collections,random,numpy as np
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
import astra_continuous as AC
from cotswitch import graders as G
D='/home/john/code/results/nocotrl-nemo/astra108/'
LC='/home/john/code/tlg-inksm/cot-control/results/outctl/length_control.py'
src=open(LC).read();src=src[src.index('BINS = '):src.index('B = rows("base")')];ns=dict(collections=collections,random=random,os=os,json=json,AC=AC);exec(src,ns)
def load(lab):
  rows=[json.loads(l) for l in open(D+lab+'.jsonl')];d=collections.defaultdict(list);lc=[]
  for r in rows:
    if r['constraint']=='ignore_question':continue
    s=G.grade('astra',r['constraint'],r['cot'],r['grader_args'])['strict'] if r['cot'] and r['closed'] else 0
    p=AC.score(dict(constraint=r['constraint'],reasoning=r['cot'],grader_args=r['grader_args']),s) if r['cot'] else 0.0
    d[r['key']].append((float(s or 0),p,len(r['cot'].split()),r['constraint']))
    lc.append(dict(item=('a',r['key']),w=len(r['cot'].split()),s=float(s or 0),c=p))
  return rows,d,lc
b_,n_=sys.argv[1],sys.argv[2];(rb,b,lb),(rn,n,ln)=load(b_),load(n_);K=sorted(set(b)&set(n));rng=np.random.default_rng(14)
for lab,rows,d in ((b_,rb,b),(n_,rn,n)):
  print(f"{lab:12} strict {100*np.mean([x[0] for k in K for x in d[k]]):5.1f}%  partial {100*np.mean([x[1] for k in K for x in d[k]]):5.1f}%  median CoT words {int(np.median([x[2] for k in K for x in d[k]]))}  truncated {sum(r['stop']=='length' for r in rows)}/{len(rows)}  unclosed {sum(not r['closed'] for r in rows)}  accuracy {100*np.mean([r['correct'] for r in rows]):.1f}%")
for i,m in ((0,'strict'),(1,'partial')):
  x=np.array([np.mean([t[i] for t in n[k]])-np.mean([t[i] for t in b[k]]) for k in K]);bs=rng.choice(x,(10000,len(x))).mean(1)
  print(f"  d {m}: {100*x.mean():+.1f} [{100*np.quantile(bs,.025):+.1f}, {100*np.quantile(bs,.975):+.1f}] (n={len(K)} items)")
for lab,lc in (('strict',None),('partial',None)):
  B=[dict(r,s=r['c']) for r in lb] if lab=='partial' else lb;R=[dict(r,s=r['c']) for r in ln] if lab=='partial' else ln
  s=ns['summary'](R,B);ci=ns['boot'](R,B,1000);f=lambda k:f"{100*s[k]:+.1f} [{100*ci[k][0]:+.1f}, {100*ci[k][1]:+.1f}]"
  print(f"  length decomposition ({lab}): raw {f('raw')} = length {f('length')} + control at same length {f('control')}")
for lab,d in ((b_,b),(n_,n)):
  by=collections.defaultdict(list)
  for k in K:
    for x in d[k]:by[x[3]].append(x)
  print(f"  {lab:12} by type strict/partial: "+' '.join(f"{c[:6]}:{int(sum(x[0] for x in v))}/{len(v)},p{100*np.mean([x[1] for x in v]):.0f}" for c,v in sorted(by.items())))
