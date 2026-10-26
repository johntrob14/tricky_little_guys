import pandas as pd,json
from pathlib import Path
p=Path(__file__).parent;f=pd.read_csv(p/'ignore_semantic_slice.csv');r=json.load(open(p/'semantic_raw134.json'));n=0
for _,x in f.iterrows():
 if pd.notna(x.first_violating_char0):
  j=int(x.first_violating_char0);assert r[int(x['index'])]['text'][j:j+len(x.violating_excerpt)]==x.violating_excerpt;n+=1
 if pd.notna(x.substantive_drift_char0):
  j=int(x.substantive_drift_char0);assert r[int(x['index'])]['text'][j:j+len(x.substantive_drift_excerpt)]==x.substantive_drift_excerpt
assert len(f)==134 and len(f[['model','item_id']].drop_duplicates())==134
print('verified excerpt offsets',n,'rows',len(f))
