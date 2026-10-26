import json,pathlib,re,pandas as pd
O=pathlib.Path(__file__).parent; B=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control');d=pd.read_csv(O/'rows.csv');cache={};supp=[];examples=[]
for r in d.to_dict('records'):
 m=r['model'];
 if m not in cache:
  z={}
  for l in open(B/'rows'/(m.replace('/','_')+'.jsonl')):
   q=json.loads(l);z[q['item_id'],q.get('salt','')]=q
  cache[m]=z
 q=cache[m][r['item_id'],''];t=q['reasoning'];a=q['content'];letters=[c for c in t if c.isalpha()];aa=[c for c in a if c.isalpha()];parity=sum(c.isupper()==(i%2==0) for i,c in enumerate(letters))/len(letters) if letters else None
 alpha=re.findall('[a-zA-Z]+',t);longwords=[w for w in alpha if len(w)>1];altmulti=sum(all(x.isupper()!=y.isupper() for x,y in zip(w,w[1:])) for w in longwords)/len(longwords) if longwords else None
 rs=dict(model=m,item_id=r['item_id'],mode=r['mode'],global_alternating_letter_share=max(parity,1-parity) if parity is not None else None,alt_multiletter_word_share=altmulti,single_letter_token_share=sum(len(w)==1 for w in alpha)/len(alpha) if alpha else None,answer_letters=len(aa),substantive_answer_only=bool(r['answer_only'] and len(aa)>=50))
 supp.append(rs)
 if r['correct_prefix_chars']>=100 and r['reference']==0 or (r['mode']=='alternating_case' and (r['score']>=.95 or rs['substantive_answer_only'])):
  p=int(r['first_violation_char0']) if pd.notna(r['first_violation_char0']) else 0;examples.append(dict(**r,head=t[:200],break_excerpt=t[max(0,p-70):p+130],tail=t[-180:],answer_head=a[:180]))
pd.DataFrame(supp).to_csv(O/'supplement.csv',index=False)
with open(O/'selected_examples.jsonl','w') as f:
 for r in examples:f.write(json.dumps(r)+'\n')
print('substantive answer only',pd.DataFrame(supp).groupby('mode').substantive_answer_only.sum().to_dict())
print('cap among passes',d[d.reference==1].groupby('mode').cap.sum().to_dict())
print('witness',d.witness.value_counts().to_dict())
print('noticed',d.groupby('mode').notice_heuristic.sum().to_dict())
print('prefix100 nonpass',d[(d.correct_prefix_chars>=100)&(d.reference==0)].groupby('mode').size().to_dict())
print('global alt exact',((d['mode']=='alternating_case')&(d.literal_global_pair_score==1)).sum())
for e in examples:
 if e['correct_prefix_chars']>=3000:print(e['model'],e['item_id'],'char',e['first_violation_char0'],'sentence',e['first_violation_sentence1'],'score',e['score'],'break',repr(e['break_excerpt']))
