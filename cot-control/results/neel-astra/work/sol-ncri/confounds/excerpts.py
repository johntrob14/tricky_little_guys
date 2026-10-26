import json,pathlib
P=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control/rows');o=[]
for slug in ['qwen_qwen3.5-397b-a17b','qwen_qwen3.5-9b','qwen_qwen3.5-27b']:
 for r in map(json.loads,open(P/(slug+'.jsonl'))):
  if r['mode']=='lowercase_thinking':
   t=r['reasoning']; ix=t.lower().find('constraint');ix=max(0,ix-30)
   o.append(dict(model=r['model'],item_id=r['item_id'],file=str(P/(slug+'.jsonl')),reasoning=t[ix:ix+280],content=r['content'][-200:]));break
json.dump(o,open('work/sol-ncri/confounds/excerpts.json','w'),indent=2);print(json.dumps(o,indent=2))
