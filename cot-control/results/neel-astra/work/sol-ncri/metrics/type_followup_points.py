from pathlib import Path
import pandas as pd,numpy as np,json,re
from scipy.stats import rankdata,spearmanr
O=Path(__file__).parent;d=pd.read_csv(O/'items.csv');m=pd.read_csv(O/'models.csv').set_index('model');root=Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control/rows')
counts={}
for mod in m.index:
 rows={}
 for l in open(root/(mod.replace('/','_')+'.jsonl')):
  r=json.loads(l)
  if not r.get('salt',''):rows[r['item_id']]=r
 for iid,r in rows.items():counts[(mod,iid)]=len(re.findall(r"[A-Za-z]+",r['reasoning']))
d['words']=[counts[(r.model,r.item_id)] for r in d.itertuples()]
d['violations']=np.where(d['mode'].isin(['word_suppression','multiple_word_suppression']),1/d.continuous-1,np.nan)
d.loc[(d.continuous<=0)|(d.words<=0),'violations']=np.nan
d['violations_per1000words']=d.violations/d.words*1000

def partial(x,y,z):
 x=rankdata(x);y=rankdata(y);Z=np.column_stack([np.ones(len(x)),z]);x=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0];y=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0];return np.corrcoef(x,y)[0,1]
out=[]
for mode in ['end_of_sentence','multiple_word_suppression','word_suppression']:
 z=d[d['mode']==mode].groupby('model').agg(continuous=('continuous','mean'),chars=('reasoning_chars','mean'),tokens=('reasoning_tokens','mean'),words=('words','mean'),violations=('violations','mean'),violation_density=('violations_per1000words','mean')).join(m[['ncri15_2','lab']])
 lab=pd.get_dummies(z.lab,dtype=float).to_numpy();ix=np.random.default_rng(41321).integers(0,len(z),(10000,len(z)))
 for metric in ['continuous']+(['violation_density','violations'] if mode!='end_of_sentence' else []):
  x=z.ncri15_2.to_numpy();y=z[metric].to_numpy()
  for control,c in [('none',np.empty((len(z),0))),('chars',rankdata(z.chars)[:,None]),('tokens',rankdata(z.tokens)[:,None]),('words',rankdata(z.words)[:,None]),('lab',lab),('lab_and_chars',np.column_stack([lab,rankdata(z.chars)]))]:
   rho=partial(x,y,c);bs=[]
   for it in ix[:10]:
    with np.errstate(divide='ignore',invalid='ignore'):v=partial(x[it],y[it],c[it])
    if np.isfinite(v):bs.append(v)
   lo,hi=np.percentile(bs,[2.5,97.5]);out.append(dict(mode=mode,metric=metric,control=control,rho=rho,bootstrap_draws=10))
 z.to_csv(O/(mode+'_followup_models.csv'))
pd.DataFrame(out).to_csv(O/'type_followup_points.csv',index=False);print(pd.DataFrame(out).to_string(index=False))
