import json,sys,collections,statistics as st,os
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
from cotswitch import graders as G
from astra_continuous import score
M='/home/john/code/tlg-inksm/cot-control/results/outctl/more/cot/';P='/home/john/code/results/nocotrl-b-pilot/';F=P+'followup_noise_elicit/'
GA={json.loads(l)['key']:json.loads(l)['grader_args'] for l in open('/home/john/code/nocotrl-b-pilot/cot-control/results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl')}
IGN=collections.defaultdict(list)
for r in map(json.loads,open(F+'ignore_judge/verdicts.jsonl')):IGN[r['src']].append(r['strict'])
def row(name,recs):  # recs: (constraint, cot, strict_or_None)
  s=[x[2] for x in recs];c=[x[1] for x in recs if x[0]!='ignore_question']
  part=[score(dict(constraint=k,reasoning=t,grader_args=GA[key]),G.grade('astra',k,t,GA[key])['strict']) for k,t,_,key in [(x[0],x[1],x[2],x[3]) for x in recs] if k!='ignore_question']
  ig=[x[2] for x in recs if x[0]=='ignore_question']
  return name,len(recs)/108,sum(s)/len(s)*108,sum(x[2] for x in recs if x[0]!='ignore_question')/max(1,len(c))*100,sum(ig)/len(ig)*100,st.mean(part)*100,st.median(len(x[1].split()) for x in recs)
out=[];agree=[0,0]
for m in sorted(os.listdir(M)):
  p=M+m+'/graded.json'
  if not os.path.exists(p):continue
  R=json.load(open(p))
  if len(R)<108:continue
  recs=[]
  for r in R:
    b=r.get('body') or '';s=int(str(r['strict']) in('1','True'))
    if r['constraint']!='ignore_question':
      g=G.grade('astra',r['constraint'],b,GA[r['key']])['strict'];agree[0]+=g==s;agree[1]+=1
    recs.append((r['constraint'],b,s,r['key']))
  out.append(row(m,recs))
def mine(src,fs):
  recs=[];ig=iter(IGN[src])
  for f in fs:
    for l in open(f):
      r=json.loads(l)
      if r['constraint']=='ignore_question':continue
      recs.append((r['constraint'],r['cot'],int(G.grade('astra',r['constraint'],r['cot'],GA[r['key']])['strict']),r['key']))
  # ignore_question verdicts (cot text not needed for partial)
  for v in IGN[src]:recs.append(('ignore_question','x'*0,v,'astra|gpqa|421|ignore_question|0'))
  return row('B '+src.split('B ')[1],recs)
out.append(mine('B base',[P+'astra_full_base.jsonl',F+'noise_base.jsonl']))
out.append(mine('B add s36',[P+'additive/astra_full_s36.jsonl',F+'noise_s36.jsonl']))
out.append(mine('B old s60',[P+'astra_full_s60.jsonl',F+'draws_old_s60/noise_old_s60.jsonl']))
out.append(mine('B add s60',[P+'additive/astra_full_s60.jsonl',F+'draws_add_s60/noise_add_s60.jsonl']))
import os as _o
for e in (72,80,90,100):
  if _o.path.exists(F+f'draws_add_s{e}/noise_add_s{e}.jsonl') and IGN.get(f'B add s{e}'):out.append(mine(f'B add s{e}',[P+f'additive/astra_full_s{e}.jsonl',F+f'draws_add_s{e}/noise_add_s{e}.jsonl']))
for v in ('fresh','stacked'):
  if _o.path.exists(P+f'variants/{v}/draws/noise_{v}_s102.jsonl') and IGN.get(f'B {v} s102'):out.append(mine(f'B {v} s102',[P+f'variants/{v}/astra_full_s102.jsonl',P+f'variants/{v}/draws/noise_{v}_s102.jsonl']))
print('grader agreement with stored strict (8 rule types):',agree)
print(f"{'run':22}{'draws':>6}{'pass/108':>9}{'rule 96 %':>10}{'ignore %':>9}{'partial %':>10}{'medW':>6}")
for n,d,p,r,i,pa,w in sorted(out,key=lambda x:-x[2]):print(f"{n:22}{d:6.0f}{p:9.1f}{r:10.1f}{i:9.0f}{pa:10.1f}{w:6.0f}")
