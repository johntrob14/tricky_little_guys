import json,pandas as pd
from pathlib import Path
O=Path(__file__).parent;B=Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control');f=pd.read_csv(O/'rows.csv');f=f[(f['mode']=='ignore_question')&(f.stored_compliance==1)]
for _,x in f.iterrows():
 rr={}
 for l in open(B/'rows'/(x.model.replace('/','_')+'.jsonl')):
  r=json.loads(l);rr[r['item_id'],r.get('salt','')]=r
 t=rr[x.item_id,'']['reasoning'];print(x.model,x.item_id,len(t));print(t if len(t)<3500 else t[:1800]+'[...]'+t[-1400:]);print()
