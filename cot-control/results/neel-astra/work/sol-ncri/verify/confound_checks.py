import pandas as pd,numpy as np,json
from scipy.stats import rankdata,spearmanr
from pathlib import Path
O=Path(__file__).parent;m=pd.read_csv(O/'models.csv');d=pd.read_csv(O/'items.csv');m['lab']=m.model.str.split('/').str[0]
c=pd.read_csv(O.parent/'confounds'/'models.csv').set_index('model');a=m.set_index('model');diff={col:float((a[col]-c[col]).abs().max()) for col in ['n','chars','tokens','accuracy','cap','empty','score']};print('raw confound differences',diff)
def pc(covs):
 X=np.column_stack([np.ones(len(m))]+[rankdata(m[k]) for k in covs]);x=rankdata(m.old_ncri);y=rankdata(m.score);return np.corrcoef(x-X@np.linalg.lstsq(X,x,rcond=None)[0],y-X@np.linalg.lstsq(X,y,rcond=None)[0])[0,1]
out={'partial':{','.join(cs):pc(cs) for cs in [['chars'],['tokens'],['accuracy'],['chars','accuracy'],['cap','empty']]},'within':{},'leave_out':{}}
for lab,f in m.groupby('lab'):
 if len(f)>1:out['within'][lab]={'n':len(f),'rho':spearmanr(f.old_ncri,f.score).statistic}
 f=m[m.lab!=lab];out['leave_out'][lab]=spearmanr(f.old_ncri,f.score).statistic
X=pd.get_dummies(m.lab).values.astype(float);x=rankdata(m.old_ncri);y=rankdata(m.score);out['lab_partial']=np.corrcoef(x-X@np.linalg.lstsq(X,x,rcond=None)[0],y-X@np.linalg.lstsq(X,y,rcond=None)[0])[0,1]
l=m.groupby('lab')[['old_ncri','score']].mean();out['between']=dict(n=len(l),rho=spearmanr(l.old_ncri,l.score).statistic)
sl=[]
for mode,f in d.groupby('mode'):
 x=np.log2(f.chars+1);xr=x-x.groupby(f.model).transform('mean');yr=f.compliance-f.groupby('model').compliance.transform('mean');sl.append(dict(mode=mode,slope=float((xr*yr).sum()/(xr*xr).sum())))
out['type_slopes']=sl;print(json.dumps(out,indent=2));(O/'confound_checks.json').write_text(json.dumps(out,indent=2))
