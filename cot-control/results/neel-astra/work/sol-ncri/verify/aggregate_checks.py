import pandas as pd,numpy as np,json
from scipy.stats import spearmanr,rankdata
from pathlib import Path
O=Path(__file__).parent;m=pd.read_csv(O/'models.csv');d=pd.read_csv(O/'items.csv');out={}
for n,f in [('full33',m[m.n==108]),('no_qwen35',m[~m.model.str.contains('qwen3.5')]),('no_glm',m[~m.model.str.startswith('z-ai/')]),('no_deepseek',m[~m.model.str.startswith('deepseek/')])]:out[n]={'rho':spearmanr(f.old_ncri,f.score).statistic,'k':int(f.pass_n.sum()),'n_items':int(f.n.sum())}
keys=set(d[d.model=='z-ai/glm-4.5'].item_id);q=d[d.item_id.isin(keys)].groupby('model').compliance.mean();out['common54']={'rho':spearmanr(m.set_index('model').loc[q.index,'old_ncri'],q).statistic,'k':int(d[d.item_id.isin(keys)].compliance.sum()),'n_items':len(d[d.item_id.isin(keys)])}
f=pd.read_csv(O/'cell_slope_checks.csv');out['cell_slopes']={'max_abs_diff':float(abs(f.slope_own-f.slope).max()),'informative':int(f.informative.sum()),'negative':int(((f.slope_own<0)&f.informative).sum()),'positive':int(((f.slope_own>0)&f.informative).sum())}
out['summary_agreement']=max(abs(m.set_index('model').score-pd.Series({s['model']:s['score'] for s in json.load(open('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control/summary.json'))})))
# Paired bootstrap rank partials using direct covariance formula, independent implementation.
rng=np.random.default_rng(523);ix=rng.integers(0,len(m),(10000,len(m)))
def corr(a,b):
 a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True);return (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))
x=rankdata(m.old_ncri.to_numpy()[ix],axis=1);y=rankdata(m.score.to_numpy()[ix],axis=1);xy=corr(x,y)
for col in ['chars','tokens','accuracy']:
 z=rankdata(m[col].to_numpy()[ix],axis=1);xz=corr(x,z);yz=corr(y,z);partial=(xy-xz*yz)/np.sqrt((1-xz*xz)*(1-yz*yz));out['partial_'+col+'_ci']=np.nanpercentile(partial,[2.5,97.5]).tolist()
print(json.dumps(out,indent=2));(O/'aggregate_checks.json').write_text(json.dumps(out,indent=2))
# Manual audit decisions after reading seed912 sample prefixes, suffixes and full ambiguous traces.
s=[r for r in json.load(open(O/'samples.json')) if r['category']=='ignore_judge'];audit=[]
for i,r in enumerate(s):audit.append({'sample_index':i,'model':r['model'],'item_id':r['item_id'],'stored':r['compliance'],'lead_assessed':0,'reason':'Empty trace' if not r['reasoning'] else ('No unrelated thought; plans answer, despite stored positive' if i==3 else 'Explicit question interpretation/planning or solution; sample8 full trace inspected')})
(O/'judge_audit.json').write_text(json.dumps({'seed':912,'agreement':29,'n':30,'reviewer':'sol-ncri AI lead, not independent human','labels':audit},indent=2))
