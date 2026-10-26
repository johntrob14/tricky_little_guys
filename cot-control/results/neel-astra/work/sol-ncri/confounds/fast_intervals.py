import pandas as pd,numpy as np,json
from scipy.stats import rankdata
from pathlib import Path
O=Path('work/sol-ncri/confounds');m=pd.read_csv(O/'models.csv');rng=np.random.default_rng(260926);B=5000;idx=rng.integers(0,len(m),(B,len(m)))
def corr(a,b):
 a=a-a.mean(axis=-1,keepdims=True);b=b-b.mean(axis=-1,keepdims=True);return (a*b).sum(axis=-1)/np.sqrt((a*a).sum(axis=-1)*(b*b).sum(axis=-1))
x=rankdata(m.ncri.to_numpy()[idx],axis=-1);out={}
for c in ['score','chars','tokens','cap','empty','accuracy']:
 y=rankdata(m[c].to_numpy()[idx],axis=-1);out[c]={'ci':np.nanpercentile(corr(x,y),[2.5,97.5]).tolist()}
y=rankdata(m.score.to_numpy()[idx],axis=-1)
for c in ['chars','tokens','accuracy']:
 z=rankdata(m[c].to_numpy()[idx],axis=-1);zc=z-z.mean(axis=-1,keepdims=True);xc=x-x.mean(axis=-1,keepdims=True);yc=y-y.mean(axis=-1,keepdims=True);den=(zc*zc).sum(axis=-1,keepdims=True);rx=xc-zc*(xc*zc).sum(axis=-1,keepdims=True)/den;ry=yc-zc*(yc*zc).sum(axis=-1,keepdims=True)/den;out['partial_'+c]={'ci':np.nanpercentile(corr(rx,ry),[2.5,97.5]).tolist()}
json.dump(out,open(O/'fast_intervals.json','w'),indent=2);print(json.dumps(out,indent=2))
