import json,collections
from pathlib import Path
b=Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control')
c=collections.Counter(); bad=[]
for p in sorted((b/'rows').glob('*.jsonl')):
 d={}
 for l in p.read_text().splitlines():
  r=json.loads(l);d[(r['item_id'],r.get('salt',''))]=r
 for (i,s),r in d.items():
  if s or not i.startswith('A:'):continue
  c[(r['model'],r['status'])]+=1
  if r['status']!='ok':bad.append({k:r.get(k) for k in ['model','item_id','status','error','finish_reason']})
out=Path('/home/john/code/results/neel-astra/work/sol-crosscheck/regrade')
(out/'non_ok.json').write_text(json.dumps(bad,indent=2))
print('non_ok',len(bad));print(c)
