import json,pathlib,pandas as pd
O=pathlib.Path(__file__).parent
for l in open(O/'audit_sample.jsonl'):
 r=json.loads(l);t=r['reasoning'] or '';p=r['first_violation_char0'];p=p if p==p else None;print('\n',r['mode'],r['sample_label'],r['model'],r['item_id'],'score',r['score'],'first_bad',p,'len',len(t),'cap',r['cap']);print('HEAD:',repr(t[:420]));print('FAIL:',repr(t[max(0,int(p or 0)-70):int(p or 0)+130]));print('END:',repr(t[-180:]));print('ANSWER:',repr(r['content'][:160]))
d=pd.read_csv(O/'rows.csv'); print(d.groupby('mode').score.describe().to_string());print(d[d.attempt_heuristic & (d.reference==0)].sort_values('correct_prefix_chars',ascending=False).head(20).to_string(index=False))
