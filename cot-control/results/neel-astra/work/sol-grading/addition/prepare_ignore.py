import pandas as pd,json,re
from pathlib import Path
P=Path(__file__).parent
D=pd.read_csv(P.parent/'lead'/'rows.csv');D=D[D['mode']=='ignore_question'].sort_values(['model','item_id']).iloc[134:268]; cache={};out=[]
for _,r in D.iterrows():
 if r.source not in cache:
  dd={}
  for l in open(r.source):
   x=json.loads(l);dd[x['item_id'],x.get('salt','')]=x
  cache[r.source]=dd
 x=cache[r.source][r.item_id,''];t=x['reasoning'];out.append(dict(model=r.model,item_id=r.item_id,source=r.source,reasoning=t,prompt=x['prompt'],stored=int(r.stored)))
json.dump(out,open(P/'ignore_review.json','w'),indent=2)
for i,r in enumerate(out):
 print(i,r['model'],r['item_id'],'len',len(r['reasoning']));print('START',repr(r['reasoning'][:420]));print('TAIL',repr(r['reasoning'][-220:]))
