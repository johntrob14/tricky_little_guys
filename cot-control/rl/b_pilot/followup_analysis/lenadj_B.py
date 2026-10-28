# Length decomposition on Astra 108 only, using length_control.py's own stats/summary/boot (exec'd verbatim up to its main block).
import json,sys,os,collections,random
LC='/home/john/code/tlg-inksm/cot-control/results/outctl/length_control.py'
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
import astra_continuous as AC
from cotswitch import graders as G
src=open(LC).read();src=src[src.index('BINS = '):src.index('B = rows("base")')]
ns=dict(collections=collections,random=random,os=os,json=json,AC=AC);exec(src,ns)
P='/home/john/code/results/nocotrl-b-pilot/';F=P+'followup_noise_elicit/';M='/home/john/code/tlg-inksm/cot-control/results/outctl/more/cot/'
GA={json.loads(l)['key']:json.loads(l)['grader_args'] for l in open('/home/john/code/nocotrl-b-pilot/cot-control/results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl')}
IGN=collections.defaultdict(list)
for r in map(json.loads,open(F+'ignore_judge/verdicts.jsonl')):IGN[r['src']].append(r)
def theirs(m,excl=()):
  out=[]
  for r in json.load(open(M+m+'/graded.json')):
    if r['constraint'] in excl:continue
    st=str(r['strict']) in('1','True');b=r.get('body') or ''
    out.append(dict(item=('more',r['key']),w=len(b.split()),s=float(st),c=AC.score(dict(constraint=r['constraint'],reasoning=b,grader_args=GA[r['key']]),st)))
  return out
def mine(src,fs,excl=()):
  out=[]
  for f in fs:
    for l in open(f):
      r=json.loads(l)
      if r['constraint']=='ignore_question' or r['constraint'] in excl:continue
      st=G.grade('astra',r['constraint'],r['cot'],r['grader_args'])['strict']==1
      out.append(dict(item=('more',r['key']),w=len(r['cot'].split()),s=float(st),c=AC.score(dict(constraint=r['constraint'],reasoning=r['cot'],grader_args=r['grader_args']),st)))
  if 'ignore_question' not in excl:
    for v in IGN[src]:out.append(dict(item=('more',v['key']),w=v['words'],s=float(v['strict']),c=float(v['strict'])))
  return out
for excl,lab in (((),'all 9 types'),(('ignore_question',),'8 rule-graded types')):
  B=theirs('base',excl)
  runs={'nocotrl1-s60':theirs('nocotrl1-s60',excl),'nocotrl1-s80x':theirs('nocotrl1-s80x',excl)}
  for e in (60,72,80):runs[f'B add s{e}']=mine(f'B add s{e}',[P+f'additive/astra_full_s{e}.jsonl',F+f'draws_add_s{e}/noise_add_s{e}.jsonl'],excl)
  print(f'\n== Astra 108, {lab}; base = nocotrl1 base (4 draws); strict pass-rate points, 95% CI over items')
  print(f"{'run':15}{'medW':>5}  {'raw strict gain':>22}  {'length part':>22}  {'control-at-length':>22}  {'partial AUC diff':>22}")
  for n,R in runs.items():
    s=ns['summary'](R,B);ci=ns['boot'](R,B,1000);f=lambda k:f"{100*s[k]:+5.1f} [{100*ci[k][0]:+5.1f},{100*ci[k][1]:+5.1f}]"
    print(f"{n:15}{sorted(r['w'] for r in R)[len(R)//2]:5}  {f('raw'):>22}  {f('length'):>22}  {f('control'):>22}  {f('auc_c'):>22}")
