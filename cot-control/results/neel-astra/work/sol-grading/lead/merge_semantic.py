import json,re
from pathlib import Path
import pandas as pd
H=Path(__file__).parent;R=H.parent;d=pd.read_csv(H/'rows.csv');raw={}
for p in set(d.source):
 for l in open(p):
  r=json.loads(l)
  if not r.get('salt'):raw[r['model'],r['item_id']]=r
parts=[];issues=[]
for sub in ['suppression','addition','stylistic']:
 a=pd.read_csv(R/sub/'ignore_semantic_slice.csv').rename(columns={'violation_excerpt':'violating_excerpt'})
 a['reviewer']=sub
 for i,x in a.iterrows():
  t=raw[x.model,x.item_id]['reasoning'];q=x.get('violating_excerpt');pos=x.get('first_violating_char0')
  if isinstance(q,str) and q:
   if q not in t:issues.append([sub,x.model,x.item_id,'excerpt absent'])
   if pd.notna(pos) and not t[int(pos):].startswith(q):issues.append([sub,x.model,x.item_id,'offset does not start excerpt',int(pos),t.find(q)])
  if pd.notna(pos):
   if not 0<=pos<=len(t):issues.append([sub,x.model,x.item_id,'position outside trace'])
   sent=1+len(re.findall(r'[.!?]\s+',t[:int(pos)]))
   if pd.notna(x.get('first_violating_sentence1')) and sent!=x.first_violating_sentence1:issues.append([sub,x.model,x.item_id,'sentence mismatch',sent,x.first_violating_sentence1])
 parts.append(a)
z=pd.concat(parts,ignore_index=True);assert len(z)==402 and not z.duplicated(['model','item_id']).any()
assert set(zip(z.model,z.item_id))==set(zip(d[d['mode']=='ignore_question'].model,d[d['mode']=='ignore_question'].item_id))
z['initial_review_scope']=z.review_scope;z['initial_taxonomy']=z.taxonomy
full=[]
for sub in ['suppression','addition','stylistic']:
 for rr in pd.read_csv(R/sub/'full_positive_audit.csv').to_dict('records'):
  key=rr['model'],rr['item_id'];t=raw[key]['reasoning']
  nch=rr.get('full_chars_read',rr.get('coverage_chars'));assert nch==len(t)
  mask=(z.model==key[0])&(z.item_id==key[1]);assert mask.sum()==1
  z.loc[mask,'taxonomy']='unrelated_only';z.loc[mask,'visible_notice']='yes';z.loc[mask,'visible_attempt']='yes'
  z.loc[mask,'review_scope']='full subsequent contiguous-chunk review; see full_positive_audit.csv'
  z.loc[mask,'full_positive_supplement']=str(R/sub/'full_positive_audit.csv')
  full.append(dict(model=key[0],item_id=key[1],chars=len(t),reviewer=sub))
assert len(full)==5
json.dump(full,(H/'full_positive_review_index.json').open('w'),indent=2)
# A sampled tail can certify a violation, but cannot certify its earliest position.
z['position_interpretation']='first identified violation; earliest not independently certified'
z.loc[z.taxonomy=='empty','position_interpretation']='no reasoning; no character position'
z.loc[z.taxonomy.isin(['uncertain','unrelated_only']),'position_interpretation']='no certified violating position; review scope applies'
z=z.rename(columns={'first_violating_char0':'first_identified_violation_char0','first_violating_sentence1':'first_identified_violation_sentence1'})
z.to_csv(H/'ignore_semantic_all.csv',index=False);z.groupby(['model','taxonomy']).size().rename('n').to_csv(H/'ignore_semantic_by_model.csv')
base=pd.read_csv(H/'all_rows.csv')
fields=['model','item_id','visible_notice','visible_attempt','taxonomy','first_identified_violation_char0','first_identified_violation_sentence1','violating_excerpt','review_scope','position_interpretation','reviewer']
base.merge(z[fields],on=['model','item_id'],how='left',validate='one_to_one').to_csv(H/'all_rows_with_semantics.csv',index=False)
json.dump(issues,(H/'semantic_validation_issues.json').open('w'),indent=2)
print('issues',json.dumps(issues));print(z.groupby(['reviewer','taxonomy']).size().to_string())
with (H/'semantic_lead_review.txt').open('w') as f:
 for x in z.sort_values(['model','item_id']).itertuples():
  q=x.violating_excerpt if isinstance(x.violating_excerpt,str) else ''
  t=raw[x.model,x.item_id]['reasoning']
  f.write(f'{x.model} {x.item_id} | {x.taxonomy} | pos={x.first_identified_violation_char0} | '+(q[:180] if q else t[:180]).replace('\n',' ')+'\n')
