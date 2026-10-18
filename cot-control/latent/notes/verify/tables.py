import pandas as pd, numpy as np, math
df = pd.read_pickle('/tmp/claude-1000/-home-john-claude-box/30da5316-57f9-530a-9d34-9d2a5e8d9097/scratchpad/verify/df.pkl')
def wilson(k,n,z=1.96):
    if n==0: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*(c-h),100*(c+h))
reg = df[df.grader!='llm_judge'].copy()
reg['strict']=reg['strict'].astype(float); reg['strict_think']=reg['strict_think'].astype(float)
print("== ARGON switch-latent (regex constraints)")
a = reg[(reg.cond=='switch-latent')&(reg.proto=='argon')]
inc = a[~a.degen]
t = inc.groupby('cons').agg(n=('strict','size'), strict=('strict','mean'), strict_think=('strict_think','mean'), cont=('cont','mean'), empty_think=('empty_think','mean'))
print((t*[1,100,100,100,100]).round(1).to_string())
k=int(inc.strict.sum()); n=len(inc)
print(f"POOLED strict {100*k/n:.1f}% (k={k}, n={n}, wilson {wilson(k,n)[0]:.1f}-{wilson(k,n)[1]:.1f}); continuous {100*inc.cont.mean():.1f}%; think-only strict {100*inc.strict_think.mean():.1f}%; degenerate excluded {a.degen.sum()}")
print("\n== DEFAULT paired: switch-latent vs qwen3-8b (regex constraints, same keys)")
d = reg[reg.proto=='default']
p = d.pivot_table(index=['key','cons','dataset'], columns='cond', values='strict').dropna()
pt = d.pivot_table(index=['key','cons','dataset'], columns='cond', values='strict_think').dropna()
print("paired n:", len(p))
g = p.groupby(level='cons').agg(['mean','size'])
print((g*100).round(1).to_string())
for name,pp in [('as-graded',p),('think-only',pt)]:
    cells = pp.groupby(level=['cons','dataset']).mean()
    print(f"{name}: pooled switch-latent {100*pp['switch-latent'].mean():.2f}% qwen3-8b {100*pp['qwen3-8b'].mean():.2f}%  | macro over {len(cells)} cells: switch {100*cells['switch-latent'].mean():.2f} qwen {100*cells['qwen3-8b'].mean():.2f}")
    # paired bootstrap over items
    rng=np.random.default_rng(0); diffs=(pp['switch-latent']-pp['qwen3-8b']).values
    bs=[rng.choice(diffs,len(diffs)).mean() for _ in range(2000)]
    print(f"   diff (switch-qwen) {100*diffs.mean():.2f} pp, 95% boot CI [{100*np.percentile(bs,2.5):.2f}, {100*np.percentile(bs,97.5):.2f}]")
print("\n== Empty think rates (full data)")
print((df.groupby(['cond','proto']).empty_think.mean()*100).round(1).to_string())
print("\n== Empty think by constraint, default, both conds (keys paired)")
dd=df[df.proto=='default']; keys=set(dd[dd.cond=='switch-latent'].key)&set(dd[dd.cond=='qwen3-8b'].key)
dd=dd[dd.key.isin(keys)]
print((dd.pivot_table(index='cons',columns='cond',values='empty_think',aggfunc='mean')*100).round(1).to_string())
print("\n== Accuracy (baseline rows; reference parse / lenient)")
for c in ['switch-latent','qwen3-8b']:
    b=df[(df.cond==c)&(df.proto=='default')&(df.cons=='baseline')]
    print(c, len(b), f"acc {100*b.correct.astype(bool).mean():.1f}% lenient {100*b.correct_len.astype(bool).mean():.1f}% noanswer {100*b.ans.isna().mean():.1f}% length_cap {100*(b.fin=='length').mean():.1f}%")
print("\n== Astra switch-latent")
s=reg[(reg.cond=='switch-latent')&(reg.proto=='astra')]
k=int(s.strict.sum()); n=len(s); print(f"regex-graded {k}/{n} = {100*k/n:.1f}% wilson {wilson(k,n)}; think-only {int(s.strict_think.sum())}/{n}")
print("\n== Latent use vs compliance (argon switch-latent, non-degenerate), controlling for length")
inc=inc.copy(); inc['switched']=inc.nblocks>0; inc['lenq']=pd.qcut(inc.ntok.astype(int),4,labels=False,duplicates='drop')
print(inc.groupby(['lenq','switched']).strict.agg(['mean','size']).round(3).to_string())
