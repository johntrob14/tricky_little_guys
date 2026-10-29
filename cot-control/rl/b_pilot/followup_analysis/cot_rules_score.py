"""Summarize cot_rules_eval: CoT strict/partial per model and rule, with paired bootstrap vs base over items."""
import json,collections,numpy as np
from pathlib import Path
O=Path('/home/john/code/results/nocotrl-b-pilot/cot_rules_eval');M=['base','B_s90','fresh_s114','stacked_s114','nocotrl1_s80']
D={m:[json.loads(l) for l in open(O/f'{m}.jsonl')] for m in M if (O/f'{m}.jsonl').exists()}
rules=['lower','avoid3','json','bullets','count','length','sections'];rng=np.random.default_rng(5)
print('CoT strict % by rule (n = 32 samples per rule per model)')
print(f"{'model':14}"+''.join(f'{r:>9}' for r in rules)+f"{'all':>8}{'partial':>9}{'medW':>6}{'correct':>8}")
for m,R in D.items():
  by=collections.defaultdict(list)
  for r in R:by[r['rule']].append(r['strict'])
  print(f'{m:14}'+''.join(f'{100*np.mean(by[x]):9.0f}' for x in rules)+f"{100*np.mean([r['strict'] for r in R]):8.1f}{100*np.mean([r['partial'] for r in R]):9.1f}{int(np.median([r['cot_words'] for r in R])):6}{100*np.mean([r['correct'] for r in R]):8.1f}")
base={};[base.setdefault(r['key'],[]).append(r) for r in D['base']]
for m,R in D.items():
  if m=='base':continue
  d={};[d.setdefault(r['key'],[]).append(r) for r in R];K=sorted(set(d)&set(base))
  for met in ('strict','partial'):
    x=np.array([np.mean([r[met] for r in d[k]])-np.mean([r[met] for r in base[k]]) for k in K]);bs=rng.choice(x,(10000,len(x))).mean(1)
    print(f'{m:14} {met:7} minus base: {100*x.mean():+5.1f} [{100*np.quantile(bs,.025):+5.1f},{100*np.quantile(bs,.975):+5.1f}]  (n={len(K)} items)')
