import json,re,pandas as pd
from pathlib import Path
P=Path(__file__).parent;D=pd.read_csv(P/'rows.csv');B=Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control')
print('counts',D.groupby('mode').size().to_dict(),'models',D.model.nunique(),'duplicates',D.duplicated(['model','item_id']).sum())
print('stored total',D.stored.sum(),'reference total',D.reference.sum(),'mismatches',(D.stored!=D.reference).sum())
for m in D.model.unique():
 slug=m.replace('/','_');rr={}
 for l in open(B/'rows'/f'{slug}.jsonl'):
  r=json.loads(l);rr[r['item_id'],r.get('salt','')]=r
 for _,q in D[(D.model==m)&(D.reference==1)&(D['mode']=='meow_between_words')].iterrows():
  r=rr[q.item_id,''];words=r['reasoning'].split();non=[w for w in words if w.lower().strip('.,!?:;\"\'()[]{}')!='meow']
  if len(non)<=1:print('VACUOUS',m,q.item_id,'tokens',len(words),'nonmeow',len(non),'finish',r['finish_reason'],'content_chars',len(r['content']))
print('answer strict marker',D.groupby('mode').apply(lambda z:((z.answer_score==1)&(z.answer_target_occurrences>0)&(z.reference==0)).sum()).to_dict())
print('near90fail',D[(D.score>=.9)&(D.reference==0)][['model','item_id','score','first_bad_char','first_bad_unit','finish_reason']].to_string(index=False))
