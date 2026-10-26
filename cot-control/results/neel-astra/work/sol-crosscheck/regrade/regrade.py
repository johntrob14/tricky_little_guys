from pathlib import Path
import json, sys, hashlib
import pandas as pd
ROOT=Path('/home/john/code/results/neel-astra/work/sol-crosscheck/regrade')
REF=Path('/home/john/code/cot-control-ref/CoT-Control-QA')
JOHN=Path('/home/john/code/tlg-arena-rl/cot-control/grading')
BASE=Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control')
sys.path.insert(0,str(REF)); sys.path.insert(0,str(JOHN))
# Full reference module import fails: missing tqdm. Compile exact original
# AST nodes for SIMPLE_MODES and grade_single_row only; genuine grading module
# imports unchanged. This avoids installing packages and excludes API code.
import ast
from grading import grade_cot_compliance, GRADE_FUNC_MAP
source=ast.parse((REF/'grade_compliance_csv.py').read_text())
nodes=[n for n in source.body if (isinstance(n,ast.FunctionDef) and n.name=='grade_single_row') or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SIMPLE_MODES' for t in n.targets))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(REF/'grade_compliance_csv.py'),'exec'),globals())
from cotswitch.graders import _grade_default
from cotswitch.data import load_items
from cotswitch.prompts import _default_grader_args
items = {(ds,it.item_idx):it for ds in ["gpqa","hle","mmlu_pro"] for it in load_items(ds)}
# Only invoke deterministic functions. No clients instantiated, no judge calls.
roster=[s.split('#')[0].strip() for s in (BASE/'roster.txt').read_text().splitlines() if s.split('#')[0].strip()]
allrows=[]; ignored=[]; raw_counts=[]
for model in roster:
 slug=model.replace('/','_')
 def load(path):
  d={}
  for line in path.read_text().splitlines():
   r=json.loads(line); d[(r['item_id'],r.get('salt',''))]=r
  return d
 raw=load(BASE/'rows'/f'{slug}.jsonl'); graded=load(BASE/'graded'/f'{slug}.jsonl')
 print(model,flush=True)
 raw_counts.append({'model':model,'raw_keys':len(raw),'graded_keys':len(graded)})
 for key,r in raw.items():
  if r.get('salt','') or not r['item_id'].startswith('A:') or r['status']!='ok': continue
  g=graded[key]
  if r['mode']=='ignore_question':
   ignored.append({'model':model,'item_id':r['item_id'],'compliance':g['compliance'],'empty':not r['reasoning'].strip()});continue
  ref=grade_single_row(pd.Series({'mode':r['mode'],'reasoning':r['reasoning'],'question':r['prompt']}))
  john=_grade_default(r['mode'],r['reasoning'],{'user_prompt':r['prompt']})
  it=items[(r['dataset'],int(r['item_id'].split(':')[2]))]
  args=_default_grader_args(it,r['mode'],r['prompt'])
  args={k:v for k,v in args.items() if k not in ('prompt','user_prompt')}
  # Same terminal-marker removal and strip as actual outctl/astra_more.py.
  import re
  body=re.sub(r"(<\|(return|end|call)\|>)+$", "", r['reasoning']).strip()
  john_kwargs=_grade_default(r['mode'],body,args)['strict'] if body else 0
  allrows.append({'model':model,'item_id':r['item_id'],'mode':r['mode'],'neel':g['compliance'],'reference':int(ref),'john':john['strict'],'john_kwargs':john_kwargs,'john_kwargs_agrees':john_kwargs==g['compliance'],'john_path':john['detail']['path'],'empty':not r['reasoning'].strip(),'raw_chars':len(r['reasoning']),'raw_file':str(BASE/'rows'/f'{slug}.jsonl'),'ref_agrees':int(ref)==g['compliance'],'john_agrees':john['strict']==g['compliance'],'ref_john_agrees':int(ref)==john['strict']})
df=pd.DataFrame(allrows)
df.to_csv(ROOT/'row_results.csv',index=False)
for cols,label in [(['mode'],'type'),(['model'],'model'),(['model','mode'],'model_type')]:
 df.groupby(cols).agg(n=('neel','size'),neel_pass=('neel','sum'),ref_pass=('reference','sum'),john_pass=('john','sum'),ref_agree=('ref_agrees','sum'),john_agree=('john_agrees','sum'),john_kwargs_pass=('john_kwargs','sum'),john_kwargs_agree=('john_kwargs_agrees','sum'),empty=('empty','sum')).reset_index().to_csv(ROOT/f'agreement_{label}.csv',index=False)
summary={'models':len(roster),'n':len(df),'modes':sorted(df['mode'].unique()),'neel_pass':int(df.neel.sum()),'ref_pass':int(df.reference.sum()),'john_pass':int(df.john.sum()),'ref_agree':int(df.ref_agrees.sum()),'john_agree':int(df.john_agrees.sum()),'john_kwargs_pass':int(df.john_kwargs.sum()),'john_kwargs_agree':int(df.john_kwargs_agrees.sum()),'ref_john_agree':int(df.ref_john_agrees.sum()),'empty':int(df['empty'].sum()),'ignore_question_n':len(ignored),'ignore_question_pass':sum(r['compliance']==1 for r in ignored),'ignore_question_empty':sum(r['empty'] for r in ignored),'raw_counts':raw_counts}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
(ROOT/'disagreements.json').write_text(json.dumps(df[~(df.ref_agrees & df.john_agrees & df.john_kwargs_agrees)].to_dict('records'),indent=2))
# Seeded review sample spanning positives and negatives, short quotes only.
review=[]
for label,part in [('pass',df[df.neel==1]),('fail',df[df.neel==0])]:
 for _,rr in part.sample(n=min(12,len(part)),random_state=20261026).iterrows():
  raw=load(Path(rr.raw_file)); r=raw[(rr.item_id,'')]
  review.append({'label':label,'model':rr.model,'item_id':rr.item_id,'mode':rr['mode'],'chars':len(r['reasoning']),'trace_prefix':r['reasoning'][:250],'trace_suffix':r['reasoning'][-250:],'source':rr.raw_file})
(ROOT/'seeded_trace_review.json').write_text(json.dumps(review,indent=2))
# Actual module source hashes for audit.
files=[REF/'grade_compliance_csv.py',REF/'grading.py',JOHN/'cotswitch/graders.py',BASE/'grade.py']
(ROOT/'source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='raw_counts'},indent=2))
print((ROOT/'agreement_type.csv').read_text())
