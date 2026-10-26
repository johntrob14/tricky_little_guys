import pandas as pd,numpy as np,json
from scipy.stats import rankdata
from pathlib import Path
O=Path(__file__).parent;d=pd.read_csv(O/'continuous_items.csv').merge(pd.read_csv(O/'items.csv')[['model','item_id','chars']],on=['model','item_id']);ncri=pd.read_csv(O/'models.csv').set_index('model').old_ncri;rng=np.random.default_rng(991);ix=rng.integers(0,34,(10000,34));out=[]
def center(v):return v-v.mean(1,keepdims=True)
def cor(a,b):return (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))
for mode in ['end_of_sentence','multiple_word_suppression']:
 f=d[d['mode']==mode].groupby('model').agg(score=('continuous','mean'),chars=('chars','mean')).join(ncri);labs=f.index.str.split('/').str[0].to_numpy()[ix];x=center(rankdata(f.old_ncri.to_numpy()[ix],axis=1));y=center(rankdata(f.score.to_numpy()[ix],axis=1));z=center(rankdata(f.chars.to_numpy()[ix],axis=1))
 for control in ['chars','lab','lab_and_chars']:
  a=x.copy();b=y.copy();c=z.copy()
  if 'lab' in control:
   for lab in np.unique(labs):
    mask=labs==lab;den=mask.sum(1);den[den==0]=1
    for v in [a,b,c]:v-=mask*((v*mask).sum(1)/den)[:,None]
  if 'chars' in control:
   den=(c*c).sum(1)
   for v in [a,b]:v-=c*((v*c).sum(1)/den)[:,None]
  rho=cor(a,b);out.append(dict(mode=mode,control=control,lo=float(np.nanpercentile(rho,2.5)),hi=float(np.nanpercentile(rho,97.5))))
pd.DataFrame(out).to_csv(O/'type_partial_cis.csv',index=False);print(pd.DataFrame(out).to_string(index=False))
