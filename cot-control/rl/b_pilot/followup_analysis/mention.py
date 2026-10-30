import json,re,collections
P='/home/john/code/results/nocotrl-b-pilot/'
pat=re.compile(r'requirement|constraint|lowercase|lower-case|uppercase|upper-case|all caps|alternat|meow|end each sentence|analysis channel|must not use|avoid (the )?word|not suspicious',re.I)
for lab,f in [('base',P+'astra_full_base.jsonl'),('s12',P+'additive/astra_full_s12.jsonl'),('s36',P+'additive/astra_full_s36.jsonl')]:
  R=[json.loads(l) for l in open(f)];R=[r for r in R if r['constraint']!='ignore_question']
  m=[r for r in R if pat.search(r['cot'])]
  # first 300 chars mention = planning to comply up front
  early=[r for r in R if pat.search(r['cot'][:300])]
  print(lab,f'mentions rule {len(m)}/{len(R)}, in first 300 chars {len(early)}/{len(R)}',dict(collections.Counter(r['constraint'] for r in m)))
