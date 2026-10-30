import json
from pathlib import Path
P=Path(__file__).parent;src=Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control/rows/z-ai_glm-5.1.jsonl');dd={}
for l in open(src):
 r=json.loads(l);dd[r['item_id'],r.get('salt','')]=r
out=[]
for q in ['152','54']:
 r=dd[f'A:gpqa:{q}:ignore_question',''];t=r['reasoning'];out.append(r)
 for i in range(0,len(t),12000):(P/f'positive_{q}_chunk{i//12000}.txt').write_text(f'CHARACTERS [{i},{min(i+12000,len(t))}) OF {len(t)}\n'+t[i:i+12000])
json.dump(out,open(P/'full_positive_raw.json','w'),indent=2)
print([(r['item_id'],len(r['reasoning'])) for r in out])
