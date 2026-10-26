import pandas as pd, json
from pathlib import Path
b=Path('/home/john/code/results/neel-astra/work/sol-crosscheck/regrade')
d=pd.read_csv(b/'row_results.csv')
# Independent source-level empirical spot checks, distinct from re-running graders.
case=d[d['mode'].isin(['lowercase_thinking','uppercase_thinking'])]
checks=[]
cache={}
for _,x in case.iterrows():
 if x.raw_file not in cache:
  rs=[json.loads(l) for l in Path(x.raw_file).read_text().splitlines()]
  cache[x.raw_file]={r['item_id']:r for r in rs if not r.get('salt')}
 r=cache[x.raw_file][x.item_id]; t=r['reasoning']; letters=[c for c in t if c.isalpha()]
 v=int(bool(t) and (not letters or all((c.islower() if x['mode']=='lowercase_thinking' else c.isupper()) for c in letters)))
 checks.append(v==x.neel)
print({'independent_case_rows':len(checks),'agree':sum(checks)})
(b/'independent_check.json').write_text(json.dumps({'independent_case_rows':len(checks),'agree':sum(checks)},indent=2))
