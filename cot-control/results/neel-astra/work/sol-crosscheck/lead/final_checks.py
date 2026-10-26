import json,pathlib,importlib.util,collections,sys,ast
import pandas as pd
O=pathlib.Path(__file__).parent;J=pathlib.Path('/home/john/code/tlg-arena-rl/cot-control');N=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control');P=pathlib.Path('/home/john/code/cot-control-ref/CoT-Control-QA')
out={}
out['glm']={}
for arm in ['cot','cot_low','expl']:
 rs=json.loads((J/f'results/outctl/xfull/{arm}/glm-base-or/graded.json').read_text()); rr=[r for r in rs if r['constraint']!='ignore_question'];out['glm'][arm]={'n':len(rs),'pass':sum(r['strict'] for r in rs),'program_n':len(rr),'program_pass':sum(r['strict'] for r in rr),'caps':sum(r.get('stop')=='length' for r in rs)}
sys.path.insert(0,str(J/'grading'))
from cotswitch import prompts
spec=importlib.util.spec_from_file_location('p',P/'prompts.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
out['templates_equal']=p.CONTROL_TYPE_REQUIREMENTS==prompts.CONTROL_TYPE_REQUIREMENTS
assert out['templates_equal']
# Independent census aggregation from row-level comparison artifact, checked against raw counts elsewhere.
df=pd.read_csv(O.parent/'regrade/row_results.csv')
out['regrade']={'n':len(df),'models':int(df.model.nunique()),'types':int(df['mode'].nunique()),'passes':int(df.neel.sum()),'ref_agree':int((df.neel==df.reference).sum()),'john_agree':int((df.neel==df.john).sum()),'kwargs_agree':int((df.neel==df.john_kwargs).sum())}
ss=json.loads((N/'summary.json').read_text());ncri=pd.read_csv('/home/john/code/neel-astra/raw/ncri_raw_data/models_ncri15_2.csv').set_index('slug')
out['ncri_summary_matches_csv']=all(abs(ncri.loc[s['model'].replace('/','_'),'ncri15_2']-s['ncri'])<1e-7 for s in ss)
out['roster_gptoss']=[s['model'] for s in ss if 'gpt-oss' in s['model']]
out['report_count']=len([p for p in (N.parent/'results/report').glob('*.md') if not p.name.startswith('._')])
out['missing_followup_paths']=[str(N.parent/d) for d in ['scratch_cot_confounds','scratch_cot_ratio','scratch_cot_elicit','scratch_cot_typesweep','scratch_cot_filler_kimi','scratch_cot_control_filler_deep','scratch_cot_priorturn','scratch_cot_sneaky','scratch_cot_decoy','scratch_cot_sneaky_win','scratch_cot_incentive','scratch_cot_small'] if not (N.parent/d).exists()]
(O/'final_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
