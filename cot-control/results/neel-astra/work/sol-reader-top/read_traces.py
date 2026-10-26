import json,csv,random
from pathlib import Path
B=Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control')
W=Path('/home/john/code/results/neel-astra/work/sol-reader-top')
models=['z-ai_glm-5','z-ai_glm-5.1','z-ai_glm-5.2','z-ai_glm-5.3','deepseek_deepseek-v3.2-exp','google_gemma-4-31b-it','moonshotai_kimi-k3','qwen_qwen3.8-2.4t-a95b','deepseek_deepseek-v4-pro']
selected=[]
for m in models:
 rows=[json.loads(l) for l in (B/'rows'/f'{m}.jsonl').read_text().splitlines()]
 grades={r['item_id']:r for r in map(json.loads,(B/'graded'/f'{m}.jsonl').read_text().splitlines())}
 for r in rows:r['grade']=grades[r['item_id']];r['file']=str(B/'rows'/f'{m}.jsonl')
 for i,mode in enumerate(sorted(set(r['mode'] for r in rows))):
  pool=[r for r in rows if r['mode']==mode]
  # Every mode per model; favor an existing success, then shortest complete stored text.
  pos=[r for r in pool if r['grade']['compliance']]
  if pos and min(len(r['reasoning']) for r in pos) <= 10000:pool=pos
  selected.append(min(pool,key=lambda r:len(r['reasoning'])))
(W/'selected.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False))
manifest=[{'model':r['model'],'item_id':r['item_id'],'mode':r['mode'],'compliance':r['grade']['compliance'],'chars':len(r['reasoning']),'finish_reason':r['finish_reason'],'file':r['file']} for r in selected]
(W/'manifest.json').write_text(json.dumps(manifest,indent=2))
for batch in range(9):
 out=[]
 for n,r in enumerate(selected[batch*9:(batch+1)*9],batch*9+1):
  out.append(f"\n=== TRACE {n}: {r['model']} | {r['item_id']} | pass={r['grade']['compliance']} | finish={r['finish_reason']} | provider={r.get('provider')} ===\nREQUIREMENT: {r['prompt'].split('Requirement:')[-1]}\nREASONING (FULL):\n{r['reasoning']}\nEND REASONING\nCONTENT: {r['content']}\nMETADATA: {json.dumps({k:r.get(k) for k in ['reasoning_tokens','witness','reasoning_details','fallback']},ensure_ascii=False)}")
 (W/f'batch{batch+1}.txt').write_text('\n'.join(out))
print('Selected',len(selected),'chars',sum(len(r['reasoning']) for r in selected),'passes',sum(r['grade']['compliance'] for r in selected))
for i in range(9):print(i+1,models[i],sum(len(r['reasoning']) for r in selected[i*9:i*9+9]))
for r in csv.DictReader(open('/home/john/code/neel-astra/raw/ncri_raw_data/models_ncri15_2.csv')):
 if r['slug'] in models:print(r['slug'],r['ncri15_2'],r['ncri15_2_rank'])
