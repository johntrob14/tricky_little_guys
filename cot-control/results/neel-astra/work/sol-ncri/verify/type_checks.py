import numpy as np,pandas as pd,json,pathlib,re
from scipy.stats import rankdata,spearmanr
O=pathlib.Path(__file__).parent;d=pd.read_csv(O/'continuous_items.csv').merge(pd.read_csv(O/'items.csv')[['model','item_id','chars','tokens']],on=['model','item_id']);base=pd.read_csv(O/'models.csv').set_index('model');out=[]
def corr(f,covs=[],lab=False):
 x=rankdata(f.ncri);y=rankdata(f.y);X=np.column_stack([np.ones(len(f))]+[rankdata(f[c]) for c in covs]+([pd.get_dummies(f.lab).values] if lab else []));return np.corrcoef(x-X@np.linalg.lstsq(X,x,rcond=None)[0],y-X@np.linalg.lstsq(X,y,rcond=None)[0])[0,1]
for mode in ['end_of_sentence','multiple_word_suppression']:
 f=d[d['mode']==mode].groupby('model').agg(y=('continuous','mean'),chars=('chars','mean')).join(base.old_ncri.rename('ncri'));f['lab']=f.index.str.split('/').str[0];out.append(dict(mode=mode,rho=corr(f),chars_partial=corr(f,['chars']),lab_partial=corr(f,lab=True),lab_chars_partial=corr(f,['chars'],True)))
B=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control/rows');a=d[d['mode']=='multiple_word_suppression'].set_index(['model','item_id']);rates=[]
for p in B.glob('*.jsonl'):
 for r in map(json.loads,p.read_text().splitlines()):
  key=(r['model'],r['item_id'])
  if key not in a.index or r.get('salt') or not r['reasoning'].strip():continue
  v=a.loc[key,'continuous'];count=round(1/v-1);words=len(re.findall('[A-Za-z]+',r['reasoning']));rates.append(dict(model=r['model'],density=1000*count/words,chars=len(r['reasoning'])))
f=pd.DataFrame(rates).groupby('model').agg(y=('density','mean'),chars=('chars','mean')).join(base.old_ncri.rename('ncri'));f['lab']=f.index.str.split('/').str[0];out.append(dict(mode='multi_density_higher_is_worse',rho=corr(f),chars_partial=corr(f,['chars']),lab_partial=corr(f,lab=True),lab_chars_partial=corr(f,['chars'],True)))
print(json.dumps(out,indent=2));(O/'type_checks.json').write_text(json.dumps(out,indent=2))
