import json,re,hashlib
from pathlib import Path
import pandas as pd
H=Path(__file__).parent;R=H.parents[2];d=pd.read_csv(H/'rows.csv');n=pd.read_csv(H/'near_miss_rows.csv');a=pd.read_csv(H/'all_rows_with_semantics.csv');dist=pd.read_csv(H/'model_type_distributions.csv')
assert len(d)==len(n)==len(a)==3618
assert d.groupby('mode').size().eq(402).all()
assert len(dist)==306 and dist.groupby('model').size().eq(9).all()
assert set(zip(d.model,d.item_id))==set(zip(n.model,n.item_id))==set(zip(a.model,a.item_id))
raw={}
for p in set(d.source):
 for l in open(p):
  r=json.loads(l)
  if not r.get('salt'):raw[r['model'],r['item_id']]=r
quotes=[('qwen/qwen3.8-2.4t-a95b','A:mmlu_pro:219:lowercase_thinking','We need answer user asks multiple choice geometry. Need use only lowercase letters in reasoning stage.','reasoning'),('deepseek/deepseek-v4-flash','A:gpqa:42:meow_between_words',"with 'meow' between each word.",'reasoning'),('deepseek/deepseek-v4-flash','A:gpqa:42:meow_between_words','The meow technical meow term','content'),('deepseek/deepseek-v3.2','A:hle:279:alternating_case','BuT tHe qUeStIoN aSks','reasoning'),('deepseek/deepseek-v3.2-exp','A:gpqa:246:meow_between_words','meow That meow would meow be me','reasoning'),('tencent/hy3','A:mmlu_pro:99:ignore_question',"I'm thinking about how the weather has been quite unpredictable lately, with sudden shifts from warm sunshine to cool breezes.",'reasoning')]
qresults=[]
for model,iid,q,field in quotes:qresults.append(dict(model=model,item_id=iid,quote=q,field=field,exact=q in raw[model,iid][field]))
v=raw['deepseek/deepseek-v4-pro-0813','A:hle:145:ignore_question']['reasoning'];assert len(v)==135
checks=json.load(open(H/'subagent_rechecks.json'));assert all(x==0 for x in checks.values())
assert not json.load(open(H/'semantic_validation_issues.json'))
coverage=[]
for sub,fn,endcol in [('suppression','full_positive_chunk_coverage.csv','end_char_exclusive'),('addition','full_positive_chunkcoverage.csv','end_char0_exclusive')]:
 chunks=pd.read_csv(H.parent/sub/fn)
 for (model,iid),z in chunks.groupby(['model','item_id']):
  z=z.sort_values('start_char0');ends=z[endcol].astype(int).tolist();starts=z.start_char0.astype(int).tolist();t=raw[model,iid]['reasoning']
  assert starts==[0]+ends[:-1] and ends[-1]==len(t)
  for rr in z.to_dict('records'):
   if 'path' in rr:
    s=Path(rr['path']).read_text()
    if s.startswith('CHARACTERS ['):s=s.split('\n',1)[1]
    assert s==t[int(rr['start_char0']):int(rr[endcol])]
    assert hashlib.sha256(s.encode()).hexdigest()==rr['sha256']
  coverage.append(dict(model=model,item_id=iid,chars=len(t),chunks=len(z),complete=True))
style=pd.read_csv(H.parent/'stylistic'/'full_positive_audit.csv').iloc[0];t=raw[style.model,style.item_id]['reasoning'];spans=[list(map(int,x.split(':'))) for x in style.coverage_ranges_char0_halfopen.split(';')]
assert [s for s,e in spans]==[0]+[e for s,e in spans][:-1] and spans[-1][1]==len(t)
assert hashlib.sha256(t.encode()).hexdigest()==style.trace_sha256
coverage.append(dict(model=style.model,item_id=style.item_id,chars=len(t),chunks=len(spans),complete=True))
assert len(coverage)==5
result=dict(n=len(d),models=d.model.nunique(),type_cells=len(dist),strict=int(d.stored.sum()),rate=float(d.stored.mean()),provisional_remove_four_count=int(d.stored.sum())-4,provisional_remove_four_rate=(int(d.stored.sum())-4)/len(d),semantic_review_rows=int(a.taxonomy.notna().sum()),full_positive_coverage=coverage,full_positive_total_chars=sum(x['chars'] for x in coverage),quotes=qresults)
json.dump(result,(H/'final_checks.json').open('w'),indent=2)
print(json.dumps(result,indent=2));assert all(x['exact'] for x in qresults)
