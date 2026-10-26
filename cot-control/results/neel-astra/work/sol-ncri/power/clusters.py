import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
rng=np.random.default_rng(2601027); out=[]
for n in [12,34]:
 x=stats.norm.ppf((np.arange(n)+.5)/n); xr=stats.rankdata(x); xc=xr-xr.mean(); crit=np.sqrt(stats.t.ppf(.975,n-2)**2/(stats.t.ppf(.975,n-2)**2+n-2))
 for r in [.4,.5,.6,.7,.8]:
  y=r*x+np.sqrt(1-r*r)*rng.normal(size=(15000,n)); yr=stats.rankdata(y,axis=1); yc=yr-yr.mean(axis=1,keepdims=True); rho=yc@xc/np.sqrt((xc**2).sum()*(yc**2).sum(1))
  out.append(dict(n=n,gaussian_copula_r=r,mean_spearman=rho.mean(),power=(abs(rho)>crit).mean(),critical_r=crit))
pd.DataFrame(out).to_csv('work/sol-ncri/power/cluster_sensitivity.csv',index=False); print(pd.DataFrame(out).to_string(index=False))
