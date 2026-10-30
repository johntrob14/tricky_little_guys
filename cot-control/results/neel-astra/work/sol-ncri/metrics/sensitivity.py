from pathlib import Path
import pandas as pd,numpy as np
from scipy.stats import spearmanr,rankdata
O=Path(__file__).parent;m=pd.read_csv(O/'models.csv').set_index('model');d=pd.read_csv(O/'items.csv');z=d[d['mode']!='ignore_question'].groupby('model').agg(binary_no_ignore=('compliance','mean'),continuous_no_ignore=('continuous','mean'));m=m.join(z)
n=pd.read_csv('/home/john/code/neel-astra/raw/ncri_raw_data/models_ncri15_2.csv').set_index('slug')
for col in ['ncri15_1_display','ncri15_0_display']:
 m[col]=[n.loc[s.replace('/','_'),col] for s in m.index]
print('old differences',[(c,(m[c]-m.ncri_old_summary).abs().max()) for c in ['ncri15_1_display','ncri15_0_display','ncri_display_bench']])
rng=np.random.default_rng(41321);ix=rng.integers(0,len(m),(10000,len(m)));out=[]
for c in ['ncri_old_summary','ncri15_2']:
 for y in ['binary_no_ignore','continuous_no_ignore']:
  a=m[c].to_numpy();b=m[y].to_numpy();ar=rankdata(a[ix],axis=1);br=rankdata(b[ix],axis=1);ar-=ar.mean(axis=1,keepdims=True);br-=br.mean(axis=1,keepdims=True);bs=(ar*br).sum(axis=1)/np.sqrt((ar*ar).sum(axis=1)*(br*br).sum(axis=1));ci=np.nanpercentile(bs,[2.5,97.5]);r=spearmanr(a,b);out.append(dict(ncri=c,metric=y,rho=r.statistic,p=r.pvalue,lo=ci[0],hi=ci[1]))
pd.DataFrame(out).to_csv(O/'sensitivity_correlations.csv',index=False);m.to_csv(O/'models.csv');print(pd.DataFrame(out).to_string(index=False));print(m[['n','compliant','binary','ncri_old_summary','ncri15_2']].to_string())
