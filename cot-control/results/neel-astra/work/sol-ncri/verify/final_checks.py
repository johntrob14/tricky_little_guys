import pandas as pd,numpy as np,json
from scipy.stats import rankdata
from pathlib import Path
O=Path(__file__).parent;m=pd.read_csv(O/'models.csv');m['lab']=m.model.str.split('/').str[0];rng=np.random.default_rng(3345)
def rho(x,y):
 a=rankdata(x);b=rankdata(y);a-=a.mean();b-=b.mean();return (a@b)/np.sqrt(a@a*(b@b))
l=m.groupby('lab')[['old_ncri','score']].mean();clusters=[m[m.lab==lab] for lab in l.index];bs=[];ls=[]
for _ in range(5000):
 ix=rng.integers(0,len(l),len(l));f=pd.concat([clusters[i] for i in ix]);bs.append(rho(f.old_ncri,f.score));a=l.iloc[ix];ls.append(rho(a.old_ncri,a.score))
shares=pd.read_csv(O/'provider_shares.csv').set_index('model');out={'lab_cluster_ci':np.nanpercentile(bs,[2.5,97.5]).tolist(),'between_lab_ci':np.nanpercentile(ls,[2.5,97.5]).tolist(),'provider_n':len(shares.columns),'multiroute_models':int(((shares>0).sum(axis=1)>1).sum()),'homogeneous_expected_zeros':34*(1-96/3631)**108,'pooled_rate':96/3618}
print(json.dumps(out,indent=2));(O/'final_checks.json').write_text(json.dumps(out,indent=2))
# ensure raw numeric tables agree with original and agent before finalizing.
a=pd.read_csv(O.parent/'metrics/models.csv').set_index('model');f=m.set_index('model')
for x,y in [('n','n'),('pass_n','compliant'),('score','binary'),('chars','reasoning_chars'),('tokens','reasoning_tokens'),('cap','capout'),('empty','empty'),('accuracy','accuracy')]:assert np.allclose(f[x].sort_index(),a[y].sort_index())
assert len(m)==34 and m.n.sum()==3618 and m.pass_n.sum()==96
print('All final raw-model table assertions passed')
