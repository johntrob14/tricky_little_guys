import numpy as np,pandas as pd,json
from scipy.stats import rankdata,spearmanr
from pathlib import Path
O=Path(__file__).parent;m=pd.read_csv(O/'models.csv');rng=np.random.default_rng(1026)
def r(a,b):
 a=rankdata(a,axis=-1);b=rankdata(b,axis=-1);a-=a.mean(axis=-1,keepdims=True);b-=b.mean(axis=-1,keepdims=True)
 with np.errstate(invalid='ignore'):return (a*b).sum(-1)/np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
out={}
for name,mask in [('glm_all',m.model.str.startswith('z-ai/')),('glm5',m.model.str.startswith('z-ai/glm-5')),('deepseek',m.model.str.startswith('deepseek/')),('qwen35',m.model.str.contains('qwen3.5'))]:
 s=m[mask];x=s.old_ncri.to_numpy();y=s.score.to_numpy();ix=rng.integers(0,len(s),(10000,len(s)));bs=r(x[ix],y[ix]);out[name]={'n':len(s),'rho':float(r(x,y)),'ci':np.nanpercentile(bs,[2.5,97.5]).tolist() if np.isfinite(bs).any() else None,'undefined':int(np.isnan(bs).sum())}
print(json.dumps(out,indent=2));(O/'family_checks.json').write_text(json.dumps(out,indent=2))
# validate every agent model/type cell slope from own selected raw-derived items
d=pd.read_csv(O/'items.csv');a=pd.read_csv(O.parent/'confounds/all_model_type_slopes.csv');out=[]
for (model,mode),s in d.groupby(['model','mode']):
 x=np.log2(s.chars+1);y=s.compliance;v=((x-x.mean())*(y-y.mean())).sum()/((x-x.mean())**2).sum();out.append({'model':model,'mode':mode,'slope_own':v})
f=pd.DataFrame(out).merge(a,on=['model','mode']);print('cell columns',f.columns.tolist());f.to_csv(O/'cell_slope_checks.csv',index=False)
