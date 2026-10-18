import pandas as pd, json, random
df = pd.read_pickle('/tmp/claude-1000/-home-john-claude-box/30da5316-57f9-530a-9d34-9d2a5e8d9097/scratchpad/verify/df.pkl')
B='/home/john/code/cot-control-switch/results_local/pod_snapshot/cot-control-switch/full1'
raw={}
for c,p in [('switch-latent','default'),('switch-latent','argon'),('qwen3-8b','default')]:
    for l in open(f'{B}/generations/{c}/{p}.jsonl'):
        r=json.loads(l); raw[(c,r['key'])]=r
reg=df[df.grader!='llm_judge'].copy(); reg['strict']=reg.strict.astype(float)
# degenerate share among passes
d=reg[(reg.proto=='default')]
print("default passes that are degenerate:", d[d.strict==1].groupby('cond').degen.mean().round(3).to_dict(), d[d.strict==1].groupby('cond').size().to_dict())
paired=set(d[d.cond=='switch-latent'].key)&set(d[d.cond=='qwen3-8b'].key)
dp=d[d.key.isin(paired)&~d.degen]
k2=dp.groupby('key').filter(lambda g: len(g)==2)
print("paired non-degenerate (both conds non-degen):", k2.groupby('cond').strict.agg(['mean','size']).round(4).to_dict())
print("paired non-degenerate think-only:", k2.groupby('cond').strict_think.apply(lambda s: s.astype(float).mean()).round(4).to_dict())
rng=random.Random(14)
def show(c,keys,n,label,chars=500):
    print(f"\n##### {label}")
    for k in rng.sample(sorted(keys),min(n,len(keys))):
        r=raw[(c,k)]; print(f"--- {k} rule={r['cot_rule']} ntok={r['n_output_tokens']} ans={r['answer_pred']}/{r['answer_gold']} lenient={r.get('answer_pred_lenient')}")
        t=r['raw_output']; print(t[:chars].replace('\n','⏎ '), ' ... ', t[-200:].replace('\n','⏎ '))
sw=reg[(reg.cond=='switch-latent')]
show('switch-latent', set(sw[(sw.proto=='default')&(sw.strict==1)&~sw.degen].key), 6, 'switch-latent default PASSES (non-degenerate)')
show('switch-latent', set(sw[(sw.proto=='argon')&(sw.strict==1)].key), 4, 'switch-latent argon PASSES', 400)
b=df[(df.cond=='switch-latent')&(df.cons=='baseline')&df.ans.isna()]
show('switch-latent', set(b.key), 4, 'switch-latent baseline NO ANSWER', 200)
