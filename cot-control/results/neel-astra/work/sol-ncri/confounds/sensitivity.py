import json,numpy as np,pandas as pd
from scipy.stats import spearmanr,rankdata
from pathlib import Path
O=Path('work/sol-ncri/confounds');d=pd.read_csv(O/'items.csv');m=pd.read_csv(O/'models.csv');rng=np.random.default_rng(261026)
def rho(f):return float(spearmanr(f.ncri,f.score).statistic)
def record(f):return dict(n=len(f),rho=rho(f),k=int(f.k.sum()),items=int(f.n.sum()))
out={}
for name,f in [('full33',m[m.n==108]),('no_qwen35',m[~m.model.str.contains('qwen3.5')]),('no_glm',m[m.lab!='z-ai']),('no_deepseek',m[m.lab!='deepseek'])]:out[name]=record(f)
# nested54 reconstructed independently from original deterministic per-dataset selection, using id qids in glm-4.5's selected54 (same draw in all models).
qids=set(':'.join(i.split(':')[:3]) for i in d[d.model=='z-ai/glm-4.5'].item_id)
f=d[d.item_id.map(lambda i:':'.join(i.split(':')[:3]) in qids)].groupby('model').agg(ncri=('ncri','first'),score=('compliance','mean'),n=('compliance','size'),k=('compliance','sum')).reset_index();out['common54']=record(f);f.to_csv(O/'common54.csv',index=False)
labs=m.lab.unique();draws=[]
for _ in range(5000):
 sampled=pd.concat([m[m.lab==l] for l in rng.choice(labs,len(labs),replace=True)]);draws.append(rho(sampled))
out['lab_cluster_bootstrap']={'ci':np.nanpercentile(draws,[2.5,97.5]).tolist(),'draws':len(draws),'clusters':len(labs)}
# per-model per-type raw slopes only cells containing both outcomes identify slopes.
cells=[]
for (model,mode),g in d.groupby(['model','mode']):
 if g.compliance.nunique()>1:
  x=np.log2(g.chars+1); y=g.compliance; cells.append(dict(model=model,mode=mode,n=len(g),k=int(y.sum()),slope=float(np.cov(x,y,ddof=0)[0,1]/np.var(x)),rho=float(spearmanr(x,y).statistic)))
c=pd.DataFrame(cells);c.to_csv(O/'informative_cells.csv',index=False);out['informative_cells']={'n':len(c),'negative':int((c.slope<0).sum()),'positive':int((c.slope>0).sum()),'zero':int((c.slope==0).sum())}
json.dump(out,open(O/'sensitivity.json','w'),indent=2);print(json.dumps(out,indent=2))
