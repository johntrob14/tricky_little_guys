import json,pathlib,sys
O=pathlib.Path(__file__).parent
p=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control/rows/z-ai_glm-5.jsonl');d={}
for l in open(p):
 r=json.loads(l);d[r['item_id'],r.get('salt','')]=r
r=d['A:mmlu_pro:124:ignore_question',''];t=r['reasoning'];(O/'giant_positive_trace.txt').write_text(t)
a=int(sys.argv[1]) if len(sys.argv)>1 else 0;b=int(sys.argv[2]) if len(sys.argv)>2 else len(t)
print('CHARS',len(t),'RANGE',a,b);print(t[a:b])
