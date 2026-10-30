import json,re,sys
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/grading']
from cotswitch import graders as G
P='/home/john/code/results/nocotrl-b-pilot/'
def first(t,n=1):
  s=re.split(r'(?<=[.!?:])\s+|\n+',t.strip());return ' '.join(s[:n])
for lab,f in [('base',P+'astra_full_base.jsonl'),('old-s24',P+'astra_full_pilot.jsonl'),('old-s36',P+'astra_full_s36.jsonl'),('old-s60',P+'astra_full_s60.jsonl'),('add-s12',P+'additive/astra_full_s12.jsonl'),('add-s24',P+'additive/astra_full_s24.jsonl'),('add-s36',P+'additive/astra_full_s36.jsonl')]:
  R=[json.loads(l) for l in open(f)];R=[r for r in R if r['constraint'] not in('ignore_question','repeat_sentences')]
  ok=[G.grade('astra',r['constraint'],first(r['cot']),r['grader_args'])['strict'] for r in R]
  case=[r for r,o in zip(R,ok) if r['constraint'] in('lowercase_thinking','uppercase_thinking','alternating_case','meow_between_words','end_of_sentence')]
  okc=[o for r,o in zip(R,ok) if r['constraint'] in('lowercase_thinking','uppercase_thinking','alternating_case','meow_between_words','end_of_sentence')]
  import collections; print(lab, dict(collections.Counter(r["constraint"] for r,o in zip(R,ok) if o)))
  if lab=="base": print("   e.g.", [first(r["cot"])[:80] for r,o in zip(R,ok) if o and r["constraint"] in ("alternating_case","meow_between_words","end_of_sentence","lowercase_thinking")][:6])
