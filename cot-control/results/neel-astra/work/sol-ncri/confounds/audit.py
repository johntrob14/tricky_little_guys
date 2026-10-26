import json,sys, pathlib, warnings
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
P=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control'); O=pathlib.Path('work/sol-ncri/confounds')
import ast, types, typing
ref=pathlib.Path('/home/john/code/cot-control-ref/CoT-Control-QA/run_cceval.py')
tree=ast.parse(ref.read_text()); node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='load_evaluation_data')
ns=dict(Path=pathlib.Path,List=typing.List,Dict=typing.Dict,Any=typing.Any,ast=ast,json=json,pd=pd)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(ref),'exec'),ns)
fake=types.ModuleType('run_cceval')
for k in ['load_evaluation_data','create_prompt_for_sample','get_control_value','convert_answer_to_letter']: setattr(fake,k,ns.get(k,lambda *a:None))
sys.modules['run_cceval']=fake
sys.path.insert(0,str(P)); import items
items.QA='/home/john/code/cot-control-ref/CoT-Control-QA'
ids108={x['item_id'] for x in items.build_block('A',12)};ids54={x['item_id'] for x in items.build_block('A',6)}
S={s['model']:s for s in json.load(open(P/'summary.json'))}; ncri=pd.read_csv('/home/john/code/neel-astra/raw/ncri_raw_data/models_ncri15_2.csv').set_index('slug')
D=[]; quotes=[]
for model in S:
 slug=model.replace('/','_'); rr={r['item_id']:r for r in map(json.loads,open(P/'rows'/f'{slug}.jsonl')) if not r.get('salt')}; gg={r['item_id']:r for r in map(json.loads,open(P/'graded'/f'{slug}.jsonl')) if not r.get('salt') and r.get('compliance') is not None}
 ids=ids108 if len(set(gg)&ids108)>=.95*108 else ids54
 for i in ids&set(gg)&set(rr):
  r,g=rr[i],gg[i]; text=r.get('reasoning') or ''; D.append(dict(model=model,lab=model.split('/')[0],item_id=i,mode=r['mode'],compliance=g['compliance'],chars=len(text),tokens=r.get('reasoning_tokens'),cap=int(r.get('finish_reason')=='length'),empty=int(not text.strip()),correct=int(r.get('correct') or False),provider=r.get('provider'),fallback=str(r.get('fallback')),witness=r.get('witness'),ncri=S[model]['ncri'],ncri15_2=ncri.loc[slug,'ncri15_2']))
  if 'qwen3.5' in model and len(quotes)<10 and r['mode'] in ['lowercase_thinking','repeat_sentences']:
   quotes.append(dict(model=model,item_id=i,file=str(P/'rows'/f'{slug}.jsonl'),excerpt=text[:280],chars=len(text),provider=r.get('provider')))
d=pd.DataFrame(D);d.to_csv(O/'items.csv',index=False)
m=d.groupby(['model','lab']).agg(n=('compliance','size'),k=('compliance','sum'),score=('compliance','mean'),ncri=('ncri','first'),ncri15_2=('ncri15_2','first'),chars=('chars','mean'),tokens=('tokens','mean'),cap=('cap','mean'),empty=('empty','mean'),accuracy=('correct','mean')).reset_index()
m['providers']=m.model.map(d.groupby('model').provider.apply(lambda x:json.dumps(x.value_counts().to_dict())))
m.to_csv(O/'models.csv',index=False)
rng=np.random.default_rng(260926)
def rho(x,y): return float(spearmanr(x,y).statistic)
def boot(frame,fun,B=5000):
 vals=[fun(frame.iloc[rng.integers(0,len(frame),len(frame))]) for _ in range(B)];return np.nanpercentile(vals,[2.5,97.5]).tolist()
def partial(frame,covs):
 X=np.column_stack([np.ones(len(frame))]+[rankdata(frame[c]) for c in covs]); a=rankdata(frame.ncri);b=rankdata(frame.score);return float(np.corrcoef(a-X@np.linalg.lstsq(X,a,rcond=None)[0],b-X@np.linalg.lstsq(X,b,rcond=None)[0])[0,1])
out={'n':len(m),'rho':rho(m.ncri,m.score),'correlations':{},'partial':{},'within_labs':{},'leave_lab_out':{}}
for c in ['score','chars','tokens','cap','empty','accuracy']:
 out['correlations'][c]={'rho':rho(m.ncri,m[c]),'ci':boot(m,lambda x:rho(x.ncri,x[c]))}
for cs in [['chars'],['tokens'],['accuracy'],['chars','accuracy'],['cap','empty']]:out['partial'][','.join(cs)]={'rho':partial(m,cs),'ci':boot(m,lambda x:partial(x,cs))}
# Residualize global ranks using lab fixed effects, avoiding ambiguous pooled within-lab ranks.
X=pd.get_dummies(m.lab,dtype=float).values;a=rankdata(m.ncri);b=rankdata(m.score)
out['lab_fixed_effect_rank_partial']=float(np.corrcoef(a-X@np.linalg.lstsq(X,a,rcond=None)[0],b-X@np.linalg.lstsq(X,b,rcond=None)[0])[0,1])
for lab,f in m.groupby('lab'):
 if len(f)>1:out['within_labs'][lab]={'n':len(f),'rho':rho(f.ncri,f.score),'models':f.model.tolist()}
 f=m[m.lab!=lab];out['leave_lab_out'][lab]={'n':len(f),'rho':rho(f.ncri,f.score)}
l=m.groupby('lab')[['ncri','score']].mean().reset_index();out['between_lab_means']={'n':len(l),'rho':rho(l.ncri,l.score),'ci':boot(l,lambda x:rho(x.ncri,x.score))}
# LPM slopes: pass probability points per log2(chars+1). Model fixed effects within each type.
d['log2chars']=np.log2(d.chars+1);sl=[]
for typ,f in d.groupby('mode'):
 fit=smf.ols('compliance ~ log2chars + C(model)',f).fit(cov_type='cluster',cov_kwds={'groups':f.model})
 sl.append(dict(mode=typ,n=len(f),passes=int(f.compliance.sum()),slope=float(fit.params['log2chars']),lo=float(fit.conf_int().loc['log2chars',0]),hi=float(fit.conf_int().loc['log2chars',1]),p=float(fit.pvalues['log2chars'])))
pd.DataFrame(sl).to_csv(O/'length_slopes.csv',index=False)
for formula in ['compliance ~ log2chars','compliance ~ log2chars + C(model)','compliance ~ log2chars + C(model) + C(mode)','compliance ~ log2chars + C(model) + C(item_id)']:
 fit=smf.ols(formula,d).fit(cov_type='cluster',cov_kwds={'groups':d.model});out[formula]={'slope':float(fit.params['log2chars']),'ci':fit.conf_int().loc['log2chars'].tolist(),'p':float(fit.pvalues['log2chars'])}
pd.crosstab(d.provider,d.lab).to_csv(O/'provider_lab.csv')
d.groupby(['model','mode']).agg(n=('compliance','size'),passes=('compliance','sum'),mean_chars=('chars','mean'),cap=('cap','mean'),empty=('empty','mean')).to_csv(O/'model_type.csv')
json.dump(out,open(O/'results.json','w'),indent=2);json.dump(quotes,open(O/'quotes.json','w'),indent=2)
print(json.dumps(out,indent=2));print(m[m.model.str.contains('qwen3.5')].to_string(index=False));print(pd.DataFrame(sl).to_string(index=False))
