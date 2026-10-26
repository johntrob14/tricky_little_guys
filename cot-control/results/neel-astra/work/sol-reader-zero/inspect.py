import json,pathlib
p=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control/rows')
r=json.loads(next((p/'qwen_qwen3.5-9b.jsonl').open()))
print(json.dumps(r,indent=2)[:14000])
b=json.load(open('/home/john/code/tlg-arena-rl/cot-control/results/outctl/more/cot/base/graded.json'))
print('BASE',type(b),str(b)[:5000])
