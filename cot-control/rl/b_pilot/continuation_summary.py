"""Independently recompute continuation metrics from preserved raw generations."""
import collections,json,random,statistics,sys
from pathlib import Path
P=Path('/home/john/code/results/nocotrl-b-pilot')
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl/b_pilot','/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
from analyze import independent_output,ci
from cotswitch import graders as G
from astra_continuous import score

def read(p):return [json.loads(l) for l in open(p)]
def main():
 out={'dev':{},'astra':{},'comparisons':{},'verification':{'seed':271030,'source':'raw JSONL, reference style graders, independent output parser'}}
 tasks={r['id']:r for r in read(P/'dev.jsonl')};D={};A={};audit=[]
 groups={'trained':['lowercase_thinking','multiple_word_suppression'],'related':['uppercase_thinking','alternating_case','word_suppression'],'clean_heldout':['meow_between_words','end_of_sentence','repeat_sentences'],'all':None}
 for label in ['base','pilot','s36']:
  rows=read(P/f'dev_{label}.jsonl');assert len(rows)==60 and len({r['task_id'] for r in rows})==60
  D[label]={r['task_id']:independent_output(tasks[r['task_id']],r) for r in rows}
  out['dev'][label]={}
  for group in ['constrained','all','agent']:
   ks=[k for k in D[label] if (group=='all' or group=='constrained' and tasks[k]['rule']!='none' or group=='agent' and tasks[k]['component']=='agent')]
   out['dev'][label][group]={'n':len(ks),**{m:sum(D[label][k][m] for k in ks) for m in ['strict_robust','joint_robust','correct_lenient','completed','task_pass','truncated']}}
  A[label]={};rowsa=read(P/f'astra_full_{label}.jsonl');assert len(rowsa)==108 and len({r['key'] for r in rowsa})==108
  for r in rowsa:
   s=G.grade('astra',r['constraint'],r['cot'],r['grader_args'])['strict'];part=score(dict(constraint=r['constraint'],reasoning=r['cot'],grader_args=r['grader_args']),s)
   A[label][r['key']]=dict(strict=s,partial=part,constraint=r['constraint'],words=len(r['cot'].split()),truncated=r['stop']=='length',correct=r['correct'])
  out['astra'][label]={}
  for group,cats in groups.items():
   rr=[v for v in A[label].values() if v['strict'] is not None and (cats is None or v['constraint'] in cats)]
   out['astra'][label][group]=dict(n=len(rr),strict=sum(r['strict'] for r in rr),partial=statistics.mean(r['partial'] for r in rr),median_words=statistics.median(r['words'] for r in rr),truncated=sum(r['truncated'] for r in rr))
  if label=='s36':
   for yes in [True,False]:
    rr=[r for r in rowsa if bool(A[label][r['key']]['strict'])==yes and A[label][r['key']]['strict'] is not None]
    audit += [dict(kind='astra',**r) for r in random.Random(271030).sample(rr,min(10,len(rr)))]
   for comp in ['ordinary','procedural','agent']:
    rr=[r for r in rows if r['component']==comp];audit += [dict(kind='dev',**r) for r in random.Random(271030).sample(rr,min(10,len(rr)))]
 for old in ['base','pilot']:
  out['comparisons'][old]={}
  ks=[k for k in D[old] if tasks[k]['rule']!='none']
  out['comparisons'][old]['dev']={m:ci([int(D['s36'][k][m])-int(D[old][k][m]) for k in ks]) for m in ['strict_robust','joint_robust']}
  out['comparisons'][old]['astra']={}
  for group,cats in groups.items():
   ks=[k for k,v in A[old].items() if v['strict'] is not None and (cats is None or v['constraint'] in cats)]
   out['comparisons'][old]['astra'][group]={m:ci([float(A['s36'][k][m])-float(A[old][k][m]) for k in ks]) for m in ['strict','partial']}
 active={};spent=0
 for r in read(P/'costs.jsonl'):
  if r['event']=='reserve':active[r['id']]=r['usd']
  else:active.pop(r['id']);spent+=r['usd']
 out['cost_upper']=spent+sum(active.values());assert out['cost_upper']<20
 (P/'continuation36_summary.json').write_text(json.dumps(out,indent=2))
 with open(P/'continuation36_audit.jsonl','w') as f:
  for r in audit:f.write(json.dumps(r)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
