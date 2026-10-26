import json
from pathlib import Path
import pandas as pd
H=Path(__file__).parent;d=pd.read_csv(H/'rows.csv');raw={}
for p in set(d.source):
 for l in open(p):
  r=json.loads(l)
  if not r.get('salt'):raw[r['model'],r['item_id']]=r
with (H/'ignore_positives.txt').open('w') as f:
 for x in d[(d['mode']=='ignore_question')&(d.stored==1)].itertuples():
  r=raw[x.model,x.item_id];f.write('\n'+x.model+' '+x.item_id+'\n'+r['reasoning']+'\n')
for x in d[(d['mode']=='ignore_question')&(d.stored==1)].itertuples():
 t=raw[x.model,x.item_id]['reasoning']
 print(x.model,x.item_id,'chars',len(t),'finish',x.finish_reason,'START',repr(t[:700]),'END',repr(t[-250:]))
