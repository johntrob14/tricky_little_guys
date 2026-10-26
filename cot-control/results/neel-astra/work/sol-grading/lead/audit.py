import sys,json,re,hashlib,ast,random
from pathlib import Path
import pandas as pd
BASE=Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control')
OUT=Path(__file__).parent
REF=Path('/home/john/code/cot-control-ref/CoT-Control-QA')
sys.path.insert(0,str(REF))
import grading as G
# Execute only the local pure dispatcher; avoid optional CLI imports (tqdm absent).
tree=ast.parse((REF/'grade_compliance_csv.py').read_text())
parts=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='grade_single_row']
ns=dict(pd=pd,grade_cot_compliance=G.grade_cot_compliance,GRADE_FUNC_MAP=G.GRADE_FUNC_MAP,SIMPLE_MODES={'alternating_case','lowercase_thinking','uppercase_thinking','no_lettere','ok_start_sentence','no_space'})
exec(compile(ast.Module(body=parts,type_ignores=[]),'reference_dispatcher','exec'),ns)
grade_single_row=ns['grade_single_row']
loader_tree=ast.parse((REF/'run_cceval.py').read_text())
loader_ns=dict(Path=Path,List=list,Dict=dict,Any=object,ast=ast,json=json)
exec(compile(ast.Module(body=[n for n in loader_tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_evaluation_data'],type_ignores=[]),'reference_loader','exec'),loader_ns)
nested={}
for ds,fn in dict(gpqa='gpqa_w_keyword.csv',hle='hle_w_keyword.csv',mmlu_pro='mmlu_pro_mini_w_keyword.csv').items():
 ids=sorted(x['id'] for x in loader_ns['load_evaluation_data'](REF/'datasets'/fn,False,'word_suppression'))
 random.Random(f'20260910-{ds}').shuffle(ids)
 nested[ds]=ids[:6]
json.dump(nested,(OUT/'nested54_qids.json').open('w'),indent=2)
def load(p):
 d={}
 for l in p.open():
  r=json.loads(l);d[r['item_id'],r.get('salt','')]=r
 return d
rows=[];raw={};extra=[]
for p in sorted((BASE/'rows').glob('*.jsonl')):
 gs=load(BASE/'graded'/p.name)
 for k,r in load(p).items():
  if k[1] or r['status']!='ok':continue
  g=gs.get(k)
  if not g or g.get('compliance') is None:continue
  # Main universe: all full-block rows except GLM4.5 nested first six questions per dataset.
  main=True
  if r['model']=='z-ai/glm-4.5':
   main=r['qid'] in nested[r['dataset']]
  if not main:extra.append((r['model'],r['item_id']));continue
  t=r.get('reasoning') or '';a=r.get('content') or '';mode=r['mode']
  def grade(text):
   return grade_single_row(pd.Series(dict(mode=mode,reasoning=text,question=r['prompt'])))
  rg=grade(t) if mode!='ignore_question' else None
  ag=grade(a) if mode!='ignore_question' else None
  rec={k:r.get(k) for k in ['model','item_id','mode','category','dataset','provider','witness','finish_reason','reasoning_tokens']}
  rec.update(source=str(p),stored=g['compliance'],reference=rg,answer_reference=ag,chars=len(t),answer_chars=len(a),empty=not t.strip(),cap=r.get('finish_reason')=='length',summary='summary' in str(r.get('reasoning_detail_types','')).lower(),fallback=bool(r.get('fallback')),chars_per_token=len(t)/(r.get('reasoning_tokens') or 1))
  rows.append(rec);raw[r['model'],r['item_id']]=r
df=pd.DataFrame(rows);df.to_csv(OUT/'rows.csv',index=False)
df.groupby('model').agg(n=('stored','size'),passes=('stored','sum'),empty=('empty','sum'),cap=('cap','sum'),summary=('summary','sum'),answer_pass=('answer_reference','sum')).to_csv(OUT/'models.csv')
df.groupby('mode').agg(n=('stored','size'),passes=('stored','sum'),empty=('empty','sum'),cap=('cap','sum'),answer_pass=('answer_reference','sum')).to_csv(OUT/'modes.csv')
program=df[df.reference.notna()]
facts=dict(n=len(df),models=df.model.nunique(),passes=int(df.stored.sum()),zero_models=int((df.groupby('model').stored.sum()==0).sum()),empty=int(df.empty.sum()) if False else int(df['empty'].sum()),cap=int(df.cap.sum()),summary=int(df.summary.sum()),witness=df.witness.value_counts().to_dict(),finish=df.finish_reason.value_counts().to_dict(),mismatch=int((program.reference!=program.stored).sum()),programmatic=len(program),answer_only=int(((program.answer_reference==1)&(program.reference==0)).sum()),excluded_extra=extra)
json.dump(facts,(OUT/'facts.json').open('w'),indent=2)
print(json.dumps(facts,indent=2));print(df.groupby('mode').stored.agg(['sum','count']).to_string())
# Seeded raw-trace reading sample, stratified by mode and stored label; no benchmark prompt text.
sample=pd.concat([x.sample(min(10,len(x)),random_state=20261026) for _,x in df.groupby(['mode','stored'])])
with (OUT/'reading_sample.txt').open('w') as f:
 for x in sample.itertuples():
  r=raw[x.model,x.item_id]
  f.write(f'\nMODEL {x.model} ITEM {x.item_id} STORED {x.stored} PROVIDER {x.provider} FINISH {x.finish_reason}\nCONSTRAINT '+r['prompt'].split('Requirement:')[-1]+'\nREASONING\n'+r['reasoning']+'\nANSWER\n'+r['content']+'\n')
print('Sample rows',len(sample))
