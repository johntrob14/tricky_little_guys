import pandas as pd,numpy as np,json
from scipy.stats import rankdata
from pathlib import Path
O=Path(__file__).parent;m=pd.read_csv(O/'models.csv').set_index('model');d=pd.read_csv(O/'items.csv');shares=pd.crosstab(d.model,d.provider).div(d.groupby('model').size(),axis=0).reindex(m.index);shares.to_csv(O/'provider_shares.csv');rng=np.random.default_rng(3491);ix=rng.integers(0,len(m),(10000,len(m)));x=m.old_ncri.to_numpy()
def cor(a,b):
 a=rankdata(a,axis=-1);b=rankdata(b,axis=-1);a-=a.mean(axis=-1,keepdims=True);b-=b.mean(axis=-1,keepdims=True);return (a*b).sum(-1)/np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
out=[]
for p in shares:
 y=shares[p].to_numpy();bs=cor(x[ix],y[ix]);out.append(dict(provider=p,models_with_route=int((y>0).sum()),rho=float(cor(x,y)),lo=float(np.nanpercentile(bs,2.5)),hi=float(np.nanpercentile(bs,97.5)),undefined_draws=int(np.isnan(bs).sum())))
for p,y in [('dominant_share',shares.max(axis=1).to_numpy()),('provider_count',(shares>0).sum(axis=1).to_numpy())]:
 bs=cor(x[ix],y[ix]);out.append(dict(provider=p,models_with_route=34,rho=float(cor(x,y)),lo=float(np.nanpercentile(bs,2.5)),hi=float(np.nanpercentile(bs,97.5)),undefined_draws=int(np.isnan(bs).sum())))
pd.DataFrame(out).to_csv(O/'provider_correlations.csv',index=False);print(pd.DataFrame(out).to_string(index=False))
