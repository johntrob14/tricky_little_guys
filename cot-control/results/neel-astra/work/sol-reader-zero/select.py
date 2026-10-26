import json,pathlib,random,collections
root=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control')
models=[l.split('|')[1].strip() for l in (root/'table.md').read_text().splitlines() if '| 0/108 |' in l]
sel=[]
for m in models:
 rows=[json.loads(l) for l in (root/'rows'/(m.replace('/','_')+'.jsonl')).read_text().splitlines()]
 by=collections.defaultdict(list)
 for r in rows: by[r['mode']].append(r)
 shortest={k:min(v,key=lambda r:len(r['reasoning'])) for k,v in by.items()}
 modes=['ignore_question']+sorted((k for k in by if k!='ignore_question'),key=lambda k:len(shortest[k]['reasoning']))[:4]
 for k in modes: sel.append(shortest[k])
# Ensure every mode represented, add shortest missing
seen={r['mode'] for r in sel}
for k in by:
 if k not in seen: sel.append(shortest[k])
out=pathlib.Path('work/sol-reader-zero')
(out/'selected.json').write_text(json.dumps(sel,indent=2))
for i,r in enumerate(sel):
 (out/f'trace-{i:02}.txt').write_text(f"{i} {r['model']} {r['item_id']}\nREQUIREMENT: {r['prompt'].split('Requirement:')[-1]}\nMETADATA: "+json.dumps({k:v for k,v in r.items() if k not in ['prompt','reasoning','content']})+f"\nREASONING FULL:\n{r['reasoning']}\nCONTENT FULL:\n{r['content']}\n")
print('models',models,'N',len(sel),'chars',sum(len(r['reasoning']) for r in sel),'modes',collections.Counter(r['mode'] for r in sel))
print([(i,r['model'],r['mode'],len(r['reasoning'])) for i,r in enumerate(sel)])
