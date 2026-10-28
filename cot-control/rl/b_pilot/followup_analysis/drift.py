import json,statistics
P='/home/john/code/results/nocotrl-b-pilot/'
def viol(t,c):
  w=t.split()
  for i,x in enumerate(w):
    L=[ch for ch in x if ch.isalpha()]
    if L and any((ch.isupper() if c=='lowercase_thinking' else ch.islower()) for ch in L):return i,len(w),x
  return None,len(w),None
for lab,f in [('base',P+'astra_full_base.jsonl'),('old-s60',P+'astra_full_s60.jsonl'),('s36',P+'additive/astra_full_s36.jsonl')]:
  R=[json.loads(l) for l in open(f)]
  for c in ('lowercase_thinking','uppercase_thinking'):
    v=[viol(r['cot'],c) for r in R if r['constraint']==c]
    print(f"{lab:8} {c:20} first-violation word idx / total words: "+', '.join(f"{i}/{n}" for i,n,_ in v))
    if lab=='old-s60' and c=='lowercase_thinking':print('   first offending tokens:',[x for _,_,x in v])
