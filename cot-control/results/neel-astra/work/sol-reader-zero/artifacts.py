import json,pathlib
root=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control/rows');s=json.load(open('work/sol-reader-zero/selected.json'))
for m in dict.fromkeys(r['model'] for r in s):
 rs=[json.loads(l) for l in (root/(m.replace('/','_')+'.jsonl')).read_text().splitlines()]
 for r in rs:
  if r['witness']!='full':print(m,r['item_id'],r['witness'],r['finish_reason'],r['provider'],len(r['reasoning']),r['reasoning_tokens'])
 print('CAP EXAMPLE',m,next(((r['item_id'],r['reasoning'][-180:],r['content'][-100:]) for r in rs if r['finish_reason']=='length'),None))
