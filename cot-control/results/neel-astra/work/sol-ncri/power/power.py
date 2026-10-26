import json, pathlib
import numpy as np,pandas as pd
from scipy import stats,optimize,special
O=pathlib.Path('work/sol-ncri/power'); rng=np.random.default_rng(2601026)
D=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control/graded')
rows=[]
for f in sorted(D.glob('*.jsonl')):
 d=[json.loads(s) for s in f.read_text().splitlines()]; rows.append(dict(model=d[0]['model'],n=len(d),k=sum(x['compliance']==1 for x in d)))
df=pd.DataFrame(rows); df['p']=df.k/df.n; df.to_csv(O/'baseline.csv',index=False)
mean=df.k.sum()/df.n.sum(); var=df.p.var(ddof=1); noise=mean*(1-mean)/108; latentvar=(var-noise)/(1-1/108)
# logistic-normal intercept, calibrated exactly by numerical normal quantile integration
zq=stats.norm.ppf((np.arange(20000)+.5)/20000)
def intercept(sig,mu=mean): return optimize.brentq(lambda a:special.expit(a+sig*zq).mean()-mu,-20,5)
def v(sig):
 p=special.expit(intercept(sig)+sig*zq); return p.var()
sfit=optimize.brentq(lambda s:v(s)-latentvar,.001,4)
base=dict(n_models=len(df),total_n=int(df.n.sum()),total_pass=int(df.k.sum()),pooled=float(mean),zero_models=int((df.k==0).sum()),var_model_rate=float(var),binomial_noise_var=float(noise),moment_latent_var=float(latentvar),moment_reliability=float(latentvar/var),fitted_logistic_sigma=float(sfit),homogeneous_expected_zeros=float(34*(1-mean)**108),p_upper_zero_one_sided95=float(1-.05**(1/108)))
# Normal ranks of a fixed 34-model predictor; dependence via Gaussian copula.
# Test uses scipy's usual t approximation for Spearman, ties ranked by average.
R=15000; n=34
x=stats.norm.ppf((np.arange(n)+.5)/n); xr=stats.rankdata(x); xc=xr-xr.mean(); xx=np.sum(xc**2)
crit=np.sqrt(stats.t.ppf(.975,n-2)**2/(stats.t.ppf(.975,n-2)**2+n-2))
def corr(arr):
 ranks=stats.rankdata(arr,axis=1); rc=ranks-ranks.mean(axis=1,keepdims=True)
 return (rc@xc)/np.sqrt((rc*rc).sum(1)*xx)
out=[]
for sig in [.5,sfit,1.5]:
 a=intercept(sig)
 for r in [0,.2,.3,.4,.5,.6,.7,.8,.9,.95]:
  zz=r*x[None,:]+np.sqrt(1-r*r)*rng.normal(size=(R,n)); p=special.expit(a+sig*zz); k=rng.binomial(108,p)
  obs=corr(k); lat=corr(p)
  out.append(dict(sigma=sig,copula_r=r,latent_spearman=float(lat.mean()),observed_spearman=float(obs.mean()),power=float((np.abs(obs)>crit).mean()),mean_rate=float(p.mean()),expected_zero_models=float((k==0).sum(1).mean()),critical_abs_r=float(crit)))
# continuous observed-rank sensitivity, no count measurement error
for r in [0,.2,.3,.4,.45,.5,.55,.6]:
 y=r*x[None,:]+np.sqrt(1-r*r)*rng.normal(size=(R,n)); rs=corr(y)
 out.append(dict(sigma=0,copula_r=r,latent_spearman=float(rs.mean()),observed_spearman=float(rs.mean()),power=float((np.abs(rs)>crit).mean()),critical_abs_r=float(crit)))
pd.DataFrame(out).to_csv(O/'power_grid.csv',index=False)
# Effective count is illustrative: rho intraclass correlation in equal-family design effect.
base['design_effect_examples']={str(rho):dict(mean_family_size=34/12,neff=34/(1+(34/12-1)*rho)) for rho in [.25,.5,1.]}
base['spearman_critical_abs']=float(crit)
(O/'summary.json').write_text(json.dumps(base,indent=2)); print(json.dumps(base,indent=2)); print(pd.DataFrame(out).to_string(index=False))
