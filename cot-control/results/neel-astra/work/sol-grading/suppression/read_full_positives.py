from pathlib import Path
import json,sys
b=Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control/rows/z-ai_glm-5.jsonl');d={}
for l in open(b):
 r=json.loads(l);d[r['item_id'],r.get('salt','')]=r
item=sys.argv[1];start=int(sys.argv[2]);end=int(sys.argv[3]);t=d[item,'']['reasoning'];print('ITEM',item,'FULL_CHARS',len(t),'COVERAGE',start,min(end,len(t)));print(t[start:end])
