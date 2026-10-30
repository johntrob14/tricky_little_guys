exec(open('work/sol-ncri/power/power.py').read().split('out=[]')[0])
out=[]
for q in [0,.25,.3823529412]:
 a=intercept(1.,mean/(1-q))
 for r in [0,.3,.6,.9]:
  zz=r*x[None,:]+np.sqrt(1-r*r)*rng.normal(size=(R,n)); active=rng.random((R,n))>=q
  p=special.expit(a+zz)*active; k=rng.binomial(108,p); obs=corr(k)
  out.append(dict(structural_zero_fraction=q,active_copula_r=r,overall_latent_spearman=float(corr(p).mean()),observed_spearman=float(obs.mean()),power=float((abs(obs)>crit).mean()),zero_models=float((k==0).sum(1).mean())))
pd.DataFrame(out).to_csv(O/'hurdle.csv',index=False); print(pd.DataFrame(out).to_string(index=False))
