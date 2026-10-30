import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
from pathlib import Path
O=Path(__file__).parent;d=pd.read_csv(O/'continuous_items.csv');base=pd.read_csv(O/'models.csv').set_index('model');rng=np.random.default_rng(20261026);ix=rng.integers(0,34,(10000,34));out=[]
for mode in ['all','no_ignore']+sorted(d['mode'].unique()):
 f=d if mode=='all' else (d[d['mode']!='ignore_question'] if mode=='no_ignore' else d[d['mode']==mode]);s=f.groupby('model').continuous.mean().to_frame().join(base.old_ncri);x=s.old_ncri.to_numpy();y=s.continuous.to_numpy();a=rankdata(x[ix],axis=1);b=rankdata(y[ix],axis=1);a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True);bs=(a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1));out.append(dict(mode=mode,rho=spearmanr(x,y).statistic,lo=np.nanpercentile(bs,2.5),hi=np.nanpercentile(bs,97.5)))
pd.DataFrame(out).to_csv(O/'continuous_correlations.csv',index=False);print(pd.DataFrame(out).to_string(index=False))
