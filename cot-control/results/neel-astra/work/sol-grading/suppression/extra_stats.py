import pandas as pd,json
from pathlib import Path
out=Path(__file__).parent
f=pd.read_csv(out/'rows.csv');d={}
for mode,z in f.groupby('mode'):
 d[mode]=dict(n=len(z),passes=int(z.stored_compliance.sum()),empty=int(z.empty_reasoning.sum()),cap=int(z.cap_out.sum()),summary=int(z.summary_flag.sum()),notice=int(z.notice_heuristic.sum()),attempt=int(z.attempt_heuristic.sum()),first_sentence_median=z.first_violation_sentence.median(),keyword_only_pass=int((z.keyword_count==0).sum()) if mode!='ignore_question' else None,taxonomy=z.taxonomy_heuristic.value_counts().to_dict())
json.dump(d,open(out/'extra_stats.json','w'),indent=2);print(json.dumps(d,indent=2))
