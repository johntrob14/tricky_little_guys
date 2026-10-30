import json,statistics,sys,collections
from pathlib import Path
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl/b_pilot','/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
from cotswitch import graders as G
from astra_continuous import score
P=Path('/home/john/code/results/nocotrl-b-pilot');F=P/'followup_noise_elicit'
groups={'trained':['lowercase_thinking','multiple_word_suppression'],'related':['uppercase_thinking','alternating_case','word_suppression'],'heldout':['meow_between_words','end_of_sentence','repeat_sentences']}
def sc(r):
  s=G.grade('astra',r['constraint'],r['cot'],r['grader_args'])['strict']
  return None if s is None else dict(c=r['constraint'],strict=s,partial=score(dict(constraint=r['constraint'],reasoning=r['cot'],grader_args=r['grader_args']),s),correct=r['correct'],trunc=r['stop']=='length')
def summ(rows):
  A=[x for x in map(sc,rows) if x];g={k:sum(a['strict'] for a in A if a['c'] in v) for k,v in groups.items()}
  return f"strict {sum(a['strict'] for a in A):2d}/{len(A)} partial {statistics.mean(a['partial'] for a in A):.3f} [trained {g['trained']}/24 related {g['related']}/36 heldout {g['heldout']}/36] correct {sum(a['correct'] for a in A)} trunc {sum(a['trunc'] for a in A)}"
R=lambda p:[json.loads(l) for l in open(p)] if Path(p).exists() else []
print('== noise floor: each draw = 1 sample per question (orig = earlier eval)')
for lab,orig,nf in [('base',P/'astra_full_base.jsonl',F/'noise_base.jsonl'),('s36',P/'additive/astra_full_s36.jsonl',F/'noise_s36.jsonl')]:
  print(f'{lab} orig  :',summ(R(orig)))
  N=R(nf)
  for i in range(3):print(f'{lab} draw{i+1}:',summ([r for r in N if r['sample']==i]))
print('== elicitation on base (2 samples per question)')
for c in ('fewshot','prefill'):
  N=R(F/f'{c}.jsonl')
  for i in range(2):print(f'{c} draw{i+1}:',summ([r for r in N if r['sample']==i]))
  A=[x for x in map(sc,N) if x];print(f'  {c} by constraint:',dict(sorted(collections.Counter(a["c"] for a in A if a["strict"]).items())))
