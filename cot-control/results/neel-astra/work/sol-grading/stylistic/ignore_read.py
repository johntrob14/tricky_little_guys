import pandas as pd,json,pathlib,re
O=pathlib.Path(__file__).parent;d=pd.read_csv(O.parent/'lead/rows.csv');d=d[d['mode']=='ignore_question'].sort_values(['model','item_id']).iloc[268:402];cache={}
with open(O/'ignore_slice_reading.txt','w') as f:
 for k,r in enumerate(d.to_dict('records')):
  if r['source'] not in cache:
   z={}
   for l in open(r['source']):
    q=json.loads(l);z[q['item_id'],q.get('salt','')]=q
   cache[r['source']]=z
  q=cache[r['source']][r['item_id'],''];t=q['reasoning'] or '';f.write(f'\nINDEX {k} {r["model"]} {r["item_id"]} N={len(t)} LABEL={r["stored"]}\nHEAD {t[:650]!r}\nTAIL {t[-200:]!r}\n')
