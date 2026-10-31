import json,collections
from pathlib import Path
import numpy as np
p=Path('outputs/generalization_data.json');d=json.loads(p.read_text());r=[json.loads(l) for l in Path('/home/john/code/tlg-arena-rl/cot-control/results/length_eval/rows_slim.jsonl').read_text().splitlines()];out={};rng=np.random.default_rng(20261030)
for c in ['think160','think225','think310','think650','think1000']:
 t=int(c[5:]);by=collections.defaultdict(lambda:collections.defaultdict(list))
 for x in r:
  if x['cond']==c:by[x['model']][x['qid']].append(abs(x['cot_tokens']-t)<=t*.25)
 keys=sorted(by['base']);ix=rng.integers(len(keys),size=(10000,len(keys)))
 for m in ['grpo3s100','len2s60']:
  v=np.array([np.mean(by[m][k])-np.mean(by['base'][k]) for k in keys]);out[m+'/'+c]={'gain':float(v.mean()),'ci95':np.quantile(v[ix].mean(1),[.025,.975]).tolist()}
d['paired_budget_gains']=out;p.write_text(json.dumps(d,indent=2));print(json.dumps(out,indent=2))
