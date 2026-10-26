from pathlib import Path
import json
from collections import Counter
files=list(Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control/graded').glob('*.jsonl'))
c=Counter(); ns=[]; zeros=0
for f in files:
 a=[json.loads(l) for l in f.open()]; ns.append(len(a)); c.update(x['compliance'] for x in a); zeros+=all(x['compliance']==0 for x in a)
print(dict(files=len(files),n=sum(ns),sizes=Counter(ns),labels=c,zero_models=zeros))
