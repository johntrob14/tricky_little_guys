import json,pathlib,random
B=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control');rng=random.Random(912)
rows=[]
for p in (B/'rows').glob('*.jsonl'):
 gs={g['item_id']:g for g in map(json.loads,(B/'graded'/p.name).read_text().splitlines()) if not g.get('salt')}
 for r in map(json.loads,p.read_text().splitlines()):
  if not r.get('salt') and r['status']=='ok' and r['item_id'] in gs:
   r['compliance']=gs[r['item_id']]['compliance'];rows.append(r)
samples=[]
for cat,pop,n in [('positive',[r for r in rows if r['compliance']],10),('negative',[r for r in rows if not r['compliance']],10),('ignore_judge',[r for r in rows if r['mode']=='ignore_question'],30)]:
 for r in rng.sample(pop,n):
  samples.append({k:r.get(k) for k in ['model','item_id','mode','control_value','compliance','reasoning','content','correct','provider','finish_reason'] }|{'category':cat})
pathlib.Path('work/sol-ncri/verify/samples.json').write_text(json.dumps(samples,indent=2))
for r in samples:
 s=r['reasoning'];print(r['category'],r['model'],r['item_id'],'label',r['compliance'],'chars',len(s),'control',repr(r['control_value']));print(s[:800]);print('END',s[-250:]);print()
