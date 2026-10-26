import json,pathlib,collections,ast,random, numpy as np,pandas as pd
from scipy.stats import spearmanr,rankdata
B=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control')
O=pathlib.Path(__file__).parent
S=json.loads((B/'summary.json').read_text()); out=[]; items=[]
for s in S:
 slug=s['model'].replace('/','_'); gs=[json.loads(l) for l in (B/'graded'/f'{slug}.jsonl').read_text().splitlines()]; rs=[json.loads(l) for l in (B/'rows'/f'{slug}.jsonl').read_text().splitlines()]
 gd={g['item_id']:g for g in gs if not g.get('salt') and g.get('compliance') is not None}; rd={r['item_id']:r for r in rs if not r.get('salt') and r['status']=='ok'}
 # Full set except glm4.5: reconstruct nested 54 by earliest two qids per type/dataset from full-model ordered rows.
 if s['model']=='z-ai/glm-4.5':
  chosen=set()
  for ds,fn in [('gpqa','gpqa_w_keyword.csv'),('hle','hle_w_keyword.csv'),('mmlu_pro','mmlu_pro_mini_w_keyword.csv')]:
   data=pd.read_csv(pathlib.Path('/home/john/code/cot-control-ref/CoT-Control-QA/datasets')/fn)
   eligible=[i for i,r in data.iterrows() if isinstance(r.valid_keywords,str) and ast.literal_eval(r.valid_keywords)]
   random.Random(f'20260910-{ds}').shuffle(eligible)
   chosen|={(ds,str(q)) for q in eligible[:6]}
  gd={k:g for k,g in gd.items() if (g['dataset'],k.split(':')[2]) in chosen}
 rows=[]
 for k,g in gd.items():
  r=rd[k]; d=dict(model=s['model'],item_id=k,mode=g['mode'],dataset=g['dataset'],compliance=g['compliance'],correct=r['correct'],chars=len(r.get('reasoning') or ''),tokens=r.get('reasoning_tokens'),cap=r.get('finish_reason')=='length',empty=not (r.get('reasoning') or '').strip(),provider=r.get('provider')); rows.append(d);items.append(d)
 f=pd.DataFrame(rows); cells=f.groupby(['dataset','mode']).compliance.mean()
 out.append(dict(model=s['model'],old_ncri=s['ncri'],n=len(f),pass_n=int(f.compliance.sum()),score=cells.mean(),accuracy=f.correct.mean(),chars=f.chars.mean(),tokens=f.tokens.fillna(0).mean(),cap=f.cap.mean(),empty=f['empty'].mean(),providers=json.dumps(collections.Counter(f.provider))))
f=pd.DataFrame(out); n=pd.read_csv('/home/john/code/neel-astra/raw/ncri_raw_data/models_ncri15_2.csv').set_index('slug');f['new_ncri']=[n.loc[m.replace('/','_'),'ncri15_2'] for m in f.model]; f.to_csv(O/'models.csv',index=False);pd.DataFrame(items).to_csv(O/'items.csv',index=False)
rng=np.random.default_rng(723);results={}
for x in ['old_ncri','new_ncri']:
 for y in ['score','accuracy','chars','tokens','cap','empty']:
  a=f[x].to_numpy();b=f[y].to_numpy(); v=[]
  for _ in range(100):
   ix=rng.integers(0,len(a),len(a));v.append(spearmanr(a[ix],b[ix]).statistic if np.std(b[ix]) else np.nan)
  results[x+'__'+y]=dict(rho=float(spearmanr(a,b).statistic),ci=np.nanpercentile(v,[2.5,97.5]).tolist())
print(f.to_string(index=False));print(json.dumps(results,indent=2));(O/'correlations.json').write_text(json.dumps(results,indent=2))
print('total',f.n.sum(),f.pass_n.sum(),'zeros',sum(f.pass_n==0));print('summary score errors',[(s['model'],float(f.set_index('model').loc[s['model'],'score'])-s['score']) for s in S if abs(float(f.set_index('model').loc[s['model'],'score'])-s['score'])>1e-10])
