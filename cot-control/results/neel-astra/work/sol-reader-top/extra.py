import json
from pathlib import Path
B=Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control');W=Path(__file__).resolve().parent
m='z-ai_glm-5'; gs={r['item_id']:r for r in map(json.loads,(B/'graded'/f'{m}.jsonl').read_text().splitlines())}
a=[r for r in map(json.loads,(B/'rows'/f'{m}.jsonl').read_text().splitlines()) if r['mode']=='ignore_question' and gs[r['item_id']]['compliance']]
r=min(a,key=lambda r:len(r['reasoning']));r['grade']=gs[r['item_id']];r['file']=str(B/'rows'/f'{m}.jsonl')
(W/'extra.json').write_text(json.dumps(r,indent=2));t=r['reasoning']
for i in range(3):(W/f'extra-part{i+1}.txt').write_text(t[i*25000:(i+1)*25000])
print({k:r[k] for k in ['model','item_id','finish_reason','reasoning_tokens','witness','provider','grade']});print(len(t))
