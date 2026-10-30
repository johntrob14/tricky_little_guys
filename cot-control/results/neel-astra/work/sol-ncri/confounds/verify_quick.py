import pandas as pd,numpy as np,json
from scipy.stats import spearmanr,rankdata
from pathlib import Path
O=Path('work/sol-ncri/confounds');m=pd.read_csv(O/'models.csv');d=pd.read_csv(O/'items.csv');r={}
for c in ['score','chars','tokens','cap','empty','accuracy']:r[c]=spearmanr(m.ncri,m[c]).statistic
for lab,g in m.groupby('lab'):
 if len(g)>1:r[lab]=dict(n=len(g),rho=float(spearmanr(g.ncri,g.score).statistic),no_lab_rho=float(spearmanr(m[m.lab!=lab].ncri,m[m.lab!=lab].score).statistic))
for c in ['chars','tokens','accuracy']:
 x,y,z=[rankdata(m[a]) for a in ['ncri','score',c]];Z=np.column_stack([np.ones(len(z)),z]);r['partial_'+c]=np.corrcoef(x-Z@np.linalg.lstsq(Z,x,rcond=None)[0],y-Z@np.linalg.lstsq(Z,y,rcond=None)[0])[0,1]
print(json.dumps(r,indent=2));print(m[m.model.str.contains('qwen3.5')].to_string(index=False))
