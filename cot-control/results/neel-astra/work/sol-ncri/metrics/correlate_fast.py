from pathlib import Path
import numpy as np,pandas as pd,json
from scipy.stats import spearmanr,rankdata
O=Path(__file__).parent;m=pd.read_csv(O/'models.csv');df=pd.read_csv(O/'items.csv');rng=np.random.default_rng(41321);ix=rng.integers(0,len(m),(10000,len(m)))
out=[]
for x in ['ncri_old_summary','ncri15_2','ncri_display_bench']:
 for y in ['binary','continuous','reasoning_chars','reasoning_tokens','capout','empty','accuracy','provider_dominance']+[f'continuous_{mode}' for mode in df['mode'].unique()]:
  a=m[x].to_numpy();b=m[y].to_numpy();ar=rankdata(a[ix],axis=1);br=rankdata(b[ix],axis=1);ar-=ar.mean(axis=1,keepdims=True);br-=br.mean(axis=1,keepdims=True)
  with np.errstate(invalid='ignore',divide='ignore'):bs=(ar*br).sum(axis=1)/np.sqrt((ar*ar).sum(axis=1)*(br*br).sum(axis=1))
  bs=bs[np.isfinite(bs)];ci=np.percentile(bs,[2.5,97.5]) if len(bs) else [np.nan,np.nan];r=spearmanr(a,b)
  out.append(dict(ncri=x,metric=y,rho=r.statistic,p=r.pvalue,lo=ci[0],hi=ci[1],n=len(m)))
c=pd.DataFrame(out);c.to_csv(O/'correlations.csv',index=False)
s={'n_models':len(m),'n_items':len(df),'compliant':int(df.compliance.sum()),'pooled_rate':df.compliance.mean(),'zero_models':int((m.compliant==0).sum()),'counts_by_n':m.n.value_counts().to_dict(),'deterministic_rows':int((df['mode']!='ignore_question').sum()),'deterministic_disagreements':len(json.load(open(O/'regrade_disagreements.json'))),'max_abs_score_difference_original':m.difference_original_score.abs().max(),'old_ncri_equals_bench':bool(np.allclose(m.ncri_old_summary,m.ncri_display_bench)),'mode_counts':df.groupby('mode').compliance.agg(['sum','size']).to_dict()}
json.dump(s,open(O/'summary.json','w'),indent=2);print(json.dumps(s,indent=2));print(c.query("ncri=='ncri_old_summary'").to_string(index=False));print(c.query("metric=='binary'").to_string(index=False))
