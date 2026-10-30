import json,random
from pathlib import Path
b=Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control')
a=[]; grades={}
for p in sorted((b/'graded').glob('*.jsonl')):
 for l in open(p):
  g=json.loads(l);grades[g['model'],g['item_id'],g.get('salt','')]=g
for p in sorted((b/'rows').glob('*.jsonl')):
 d={}
 for l in open(p):
  r=json.loads(l);d[r['item_id'],r.get('salt','')]=r
 for (i,s),r in d.items():
  if not s and r.get('status')=='ok' and i.startswith('A:') and r['mode']=='ignore_question':a.append(r)
rng=random.Random(260926)
audit=[]
for r in rng.sample(a,30):
 hand=int(r['model']=='tencent/hy3' and r['item_id']=='A:mmlu_pro:99:ignore_question')
 audit.append(dict(model=r['model'],item_id=r['item_id'],hand_compliance=hand,stored_compliance=grades[r['model'],r['item_id'],'']['compliance'],basis='entire unrelated 491-character trace' if hand else 'empty reasoning' if not r['reasoning'] else 'visible question analysis or answer planning in inspected excerpts'))
 t=r['reasoning'];print(r['model'],r['item_id'],'chars',len(t),'\n',t[:600].replace('\n',' '),'[...]',t[-200:].replace('\n',' '),'\n')

import pandas as pd
pd.DataFrame(audit).to_csv(Path(__file__).parent/"ignore_manual30.csv",index=False)
print("AGREEMENT",sum(x["hand_compliance"]==x["stored_compliance"] for x in audit),len(audit))
