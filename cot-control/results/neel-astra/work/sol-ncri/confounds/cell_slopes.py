import pandas as pd,numpy as np,json
from scipy.stats import spearmanr
from pathlib import Path
O=Path('work/sol-ncri/confounds');d=pd.read_csv(O/'items.csv');m=pd.read_csv(O/'models.csv');out=[]
for (model,mode),g in d.groupby(['model','mode']):
 x=np.log2(g.chars+1);y=g.compliance;v=bool(y.nunique()>1 and x.nunique()>1)
 out.append(dict(model=model,mode=mode,n=len(g),passes=int(y.sum()),informative=v,slope=float(np.cov(x,y,ddof=0)[0,1]/np.var(x)) if v else None,rho=float(spearmanr(x,y).statistic) if v else None))
pd.DataFrame(out).to_csv(O/'all_model_type_slopes.csv',index=False)
z=m[m.lab=='z-ai'];out={}
for name,g in [('glm5',z[z.model.str.contains('glm-5')]),('glm4',z[z.model.str.contains('glm-4')]),('qwen35',m[m.model.str.contains('qwen3.5')])]:
 out[name]={'n':len(g),'rho':float(spearmanr(g.ncri,g.score).statistic),'models':g[['model','ncri','score']].to_dict('records')}
json.dump(out,open(O/'family_correlations.json','w'),indent=2);print(json.dumps(out,indent=2))
