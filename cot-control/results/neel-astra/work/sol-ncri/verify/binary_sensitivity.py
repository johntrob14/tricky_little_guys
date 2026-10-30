import numpy as np,pandas as pd,json
from scipy.stats import spearmanr,rankdata
from pathlib import Path
O=Path(__file__).parent;d=pd.read_csv(O/'items.csv');m=pd.read_csv(O/'models.csv').set_index('model');f=d[d['mode']!='ignore_question'].groupby('model').compliance.mean().to_frame().join(m.old_ncri);x=f.old_ncri.to_numpy();y=f.compliance.to_numpy();ix=np.random.default_rng(615).integers(0,34,(10000,34));a=rankdata(x[ix],axis=1);b=rankdata(y[ix],axis=1);a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True);r=(a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1));out=dict(rho=spearmanr(x,y).statistic,ci=np.nanpercentile(r,[2.5,97.5]).tolist());(O/'binary_no_ignore.json').write_text(json.dumps(out,indent=2));print(out)
