import pandas as pd,json,re
from pathlib import Path
O=Path(__file__).parent;f=pd.read_csv(O.parent/'lead/rows.csv');f=f[f['mode']=='ignore_question'].sort_values(['model','item_id']).reset_index(drop=True).iloc[:134];data=[]
for idx,x in f.iterrows():
 d={}
 for l in open(x.source):
  r=json.loads(l);d[r['item_id'],r.get('salt','')]=r
 r=d[x.item_id,''];data.append(dict(index=int(idx),model=x.model,item_id=x.item_id,text=r['reasoning'],source=x.source,stored=int(x.stored)))
json.dump(data,open(O/'semantic_raw134.json','w'))
import sys
start=int(sys.argv[1]) if len(sys.argv)>1 else 0;end=int(sys.argv[2]) if len(sys.argv)>2 else 134
for r in data[start:end]:
 t=r['text'];n=re.search(r'(?:instruction|constraint|requirement|must not reason|not think about|anything else|something else)',t,re.I)
 print(r['index'],r['model'],r['item_id'],'CHARS',len(t),'STORED',r['stored']);print('START',repr(t[:650]));print('END',repr(t[-200:]));
 if n:print('NOTICE',n.start(),repr(t[max(0,n.start()-70):n.end()+170]))
