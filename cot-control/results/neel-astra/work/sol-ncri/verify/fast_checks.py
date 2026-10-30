import pandas as pd,numpy as np,json,pathlib
from scipy import stats,special,optimize
O=pathlib.Path(__file__).parent;f=pd.read_csv(O/'models.csv');rng=np.random.default_rng(723);ix=rng.integers(0,len(f),(10000,len(f)))
def cor(a,b):
 a=stats.rankdata(a,axis=-1);b=stats.rankdata(b,axis=-1);a=a-a.mean(axis=-1,keepdims=True);b=b-b.mean(axis=-1,keepdims=True);return (a*b).sum(-1)/np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
out={}
for x in ['old_ncri','new_ncri']:
 for y in ['score','accuracy','chars','tokens','cap','empty']:
  a=f[x].to_numpy();b=f[y].to_numpy();c=cor(a[ix],b[ix]);out[x+'__'+y]={'rho':float(cor(a,b)),'ci':np.nanpercentile(c,[2.5,97.5]).tolist()}
(O/'fast_correlations.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));print('counts',f.n.sum(),f.pass_n.sum(),sum(f.pass_n==0));print('gauge',stats.spearmanr(f.old_ncri,f.new_ncri));
# independent power checks, fixed predictor and variance-matched logistic rates
n=34;t=stats.t.ppf(.975,32);critical=t/np.sqrt(t*t+32);x=stats.norm.ppf((np.arange(n)+.5)/n)
mean=96/3631;R=20000;power=[]
for sigma in [0,1.0603585218]:
 z=stats.norm.ppf((np.arange(30000)+.5)/30000);a=optimize.brentq(lambda a:special.expit(a+sigma*z).mean()-mean,-20,5) if sigma else 0
 for r in [.3,.5,.55,.6]:
  noise=r*x+np.sqrt(1-r*r)*rng.normal(size=(R,n));latent=noise if sigma==0 else special.expit(a+sigma*noise);observed=latent if sigma==0 else rng.binomial(108,latent);c=cor(np.broadcast_to(x,(R,n)),observed);power.append({'sigma':sigma,'r':r,'mean_observed':float(c.mean()),'power':float((abs(c)>critical).mean())})
print('power',power);(O/'power_checks.json').write_text(json.dumps({'critical':critical,'zero_upper':1-.05**(1/108),'power':power},indent=2))
