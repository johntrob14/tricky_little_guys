import json,sys,collections,re,numpy as np
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
from cotswitch import graders as G
from astra_continuous import score
P='/home/john/code/results/nocotrl-b-pilot/';F=P+'followup_noise_elicit/';M='/home/john/code/tlg-inksm/cot-control/results/outctl/more/cot/'
GA={json.loads(l)['key']:json.loads(l)['grader_args'] for l in open('/home/john/code/nocotrl-b-pilot/cot-control/results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl')}
def first(t):return re.split(r'(?<=[.!?:])\s+|\n+',t.strip())[0]
def add(d,key,c,cot):
  s=G.grade('astra',c,cot,GA[key])['strict']
  if s is None:return
  d[key].append(dict(s=s,p=score(dict(constraint=c,reasoning=cot,grader_args=GA[key]),s),c=c,w=len(cot.split()),on=G.grade('astra',c,first(cot),GA[key])['strict'] if c=='lowercase_thinking' else None))
def mine(*fs):
  d=collections.defaultdict(list)
  for f in fs:
    for l in open(f):r=json.loads(l);add(d,r['key'],r['constraint'],r['cot'])
  return d
def theirs(m):
  d=collections.defaultdict(list)
  for r in json.load(open(M+m+'/graded.json')):add(d,r['key'],r['constraint'],r['body'] or '')
  return d
mybase=mine(P+'astra_full_base.jsonl',F+'noise_base.jsonl');K=sorted(mybase);rng=np.random.default_rng(271031)
D={'B base (mine)':mybase,'nocotrl1 base':theirs('base'),
   **{f'nocotrl0 s80':theirs('nocotrl0-s80')},**{f'nocotrl1 s{s}':theirs(f'nocotrl1-s{s}') for s in (20,40,60,80)},'nocotrl1 s80x':theirs('nocotrl1-s80x'),
   'B old s60':mine(P+'astra_full_s60.jsonl',F+'draws_old_s60/noise_old_s60.jsonl'),'B add s60':mine(P+'additive/astra_full_s60.jsonl',F+'draws_add_s60/noise_add_s60.jsonl')}
pooled={k:D['B base (mine)'][k]+D['nocotrl1 base'][k] for k in K}  # 8 base draws
def ci(x):bs=rng.choice(x,(10000,len(x))).mean(1);return f'{x.mean()*100:+5.1f} [{np.quantile(bs,.025)*100:+5.1f},{np.quantile(bs,.975)*100:+5.1f}]'
tr=('lowercase_thinking','multiple_word_suppression');ho=('meow_between_words','end_of_sentence','repeat_sentences')
print(f"{'model':15}{'n':>5}{'strict':>8}{'partial':>8}  {'d partial vs 8-draw base':>24}  {'d partial trained':>22}  {'d partial clean held-out':>24} {'medW':>5} {'lc onset':>8}")
for n,d in D.items():
  a=[t for k in K for t in d[k]]
  dl=lambda keys:np.array([np.mean([t['p'] for t in d[k]])-np.mean([t['p'] for t in pooled[k]]) for k in keys])
  on=[t['on'] for t in a if t['on'] is not None]
  print(f"{n:15}{len(a):5}{sum(t['s'] for t in a)/len(a)*100:7.1f}%{np.mean([t['p'] for t in a]):8.3f}  {ci(dl(K)):>24}  {ci(dl([k for k in K if pooled[k][0]['c'] in tr])):>22}  {ci(dl([k for k in K if pooled[k][0]['c'] in ho])):>24} {int(np.median([t['w'] for t in a])):5} {sum(on):>3}/{len(on)}")
