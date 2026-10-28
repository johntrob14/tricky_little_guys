import json,statistics,sys
from pathlib import Path
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl/b_pilot','/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
from cotswitch import graders as G
from astra_continuous import score
P=Path('/home/john/code/results/nocotrl-b-pilot')
groups={'trained':['lowercase_thinking','multiple_word_suppression'],'related':['uppercase_thinking','alternating_case','word_suppression'],'clean_heldout':['meow_between_words','end_of_sentence','repeat_sentences'],'all':None}
for label,path in [('base',P/'astra_full_base.jsonl'),('old-s60',P/'astra_full_s60.jsonl')]+[(f'add-s{s}',P/f'additive/astra_full_s{s}.jsonl') for s in (12,24,36)]:
  rows=[json.loads(l) for l in open(path)];assert len(rows)==108
  A=[]
  for r in rows:
    s=G.grade('astra',r['constraint'],r['cot'],r['grader_args'])['strict']
    if s is None: continue
    A.append(dict(c=r['constraint'],strict=s,partial=score(dict(constraint=r['constraint'],reasoning=r['cot'],grader_args=r['grader_args']),s),correct=r['correct'],trunc=r['stop']=='length',words=len(r['cot'].split())))
  out=[]
  for g,cats in groups.items():
    rr=[a for a in A if cats is None or a['c'] in cats]
    out.append(f"{g}: {sum(a['strict'] for a in rr)}/{len(rr)} strict, partial {statistics.mean(a['partial'] for a in rr):.3f}")
  print(label,'|',' | '.join(out),f"| correct {sum(a['correct'] for a in A)}/{len(A)} trunc {sum(a['trunc'] for a in A)} medwords {statistics.median(a['words'] for a in A)}")
