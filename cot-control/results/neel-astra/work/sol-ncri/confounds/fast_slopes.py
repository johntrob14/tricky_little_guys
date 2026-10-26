import pandas as pd,numpy as np,json,statsmodels.formula.api as smf
from pathlib import Path
O=Path('work/sol-ncri/confounds');d=pd.read_csv(O/'items.csv');d['log2chars']=np.log2(d.chars+1);sl=[]
for typ,f in d.groupby('mode'):
 fit=smf.ols('compliance ~ log2chars + C(model)',f).fit(cov_type='cluster',cov_kwds={'groups':f.model});sl.append(dict(mode=typ,n=len(f),passes=int(f.compliance.sum()),slope=float(fit.params['log2chars']),lo=float(fit.conf_int().loc['log2chars',0]),hi=float(fit.conf_int().loc['log2chars',1]),p=float(fit.pvalues['log2chars'])))
pd.DataFrame(sl).to_csv(O/'length_slopes.csv',index=False);out={}
for formula in ['compliance ~ log2chars','compliance ~ log2chars + C(model)','compliance ~ log2chars + C(model) + C(mode)','compliance ~ log2chars + C(model) + C(item_id)']:
 fit=smf.ols(formula,d).fit(cov_type='cluster',cov_kwds={'groups':d.model});out[formula]={'slope':float(fit.params['log2chars']),'ci':fit.conf_int().loc['log2chars'].tolist(),'p':float(fit.pvalues['log2chars'])}
json.dump(out,open(O/'fast_slopes.json','w'),indent=2);print(pd.DataFrame(sl).to_string(index=False));print(json.dumps(out,indent=2))
