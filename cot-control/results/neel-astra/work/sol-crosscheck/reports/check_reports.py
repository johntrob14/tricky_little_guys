import json,pathlib,collections,re,importlib.util,sys,types
import numpy as np
from scipy.stats import spearmanr,pearsonr,rankdata,binomtest
BASE=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data')
OUT=pathlib.Path(__file__).parent
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
ss=json.loads((BASE/'scratch_cot_control/summary.json').read_text())
gs=[r for p in (BASE/'scratch_cot_control/graded').glob('*.jsonl') for r in rows(p)]
by=collections.defaultdict(list)
for r in gs:by[r['model']].append(r)
# Main published analysis truncates each dataset/type cell to the common nested item tranche.
per=[]
for s in ss:
 rr=by[s['model']]; per.append(dict(model=s['model'],n=len(rr),k=sum(r['compliance']==1 for r in rr),score_raw=np.mean([r['compliance']==1 for r in rr]),score_published=s['score'],ncri=s['ncri'],accuracy_raw=np.mean([bool(r['correct']) for r in rr])))
out={'main_raw':{'n':len(gs),'k':sum(r['compliance']==1 for r in gs),'empty':sum(r['empty_reasoning'] for r in gs),'models':len(by),'zero_models':sum(x['k']==0 for x in per)},'main_per_model':per}
out['published_summary_sums']={key:sum(s[key] for s in ss) for key in ['n_all','n_compliant_all','n_graded','n_compliant']}
out['summary_correlations']={'spearman':list(spearmanr([s['ncri'] for s in ss],[s['score'] for s in ss])),'pearson':list(pearsonr([s['ncri'] for s in ss],[s['score'] for s in ss])),'ncri_accuracy':list(spearmanr([s['ncri'] for s in ss],[s['accuracy'] for s in ss]))}
out['raw_correlations']={'spearman':list(spearmanr([s['ncri'] for s in per],[s['score_raw'] for s in per]))}
out['mode_raw']={m:{'n':len(rr),'k':sum(r['compliance']==1 for r in rr)} for m in sorted(set(r['mode'] for r in gs)) for rr in [[r for r in gs if r['mode']==m]]}
raw=[r for p in (BASE/'scratch_cot_control/rows').glob('*.jsonl') for r in rows(p)]
positive={(r['model'],r['item_id']) for r in gs if r['compliance']==1}
out['main_trace_checks']={'raw_rows':len(raw),'any_cjk':sum(bool(re.search('[\u3400-\u4dbf\u4e00-\u9fff]',r.get('reasoning') or '')) for r in raw),'positive_below20latin':sum((r['model'],r['item_id']) in positive and len(re.findall('[A-Za-z]',r.get('reasoning') or ''))<20 for r in raw)}
spec=importlib.util.spec_from_file_location('reference_grading','/home/john/code/cot-control-ref/CoT-Control-QA/grading.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
v=types.ModuleType('vendor');v.cc_grading=g;sys.modules['vendor']=v
spec=importlib.util.spec_from_file_location('cc_score',BASE/'scratch_cotcontrol/score.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
qa=rows(BASE/'scratch_cotcontrol/out/rows_qa_mvp.jsonl')
out['mvp_qa']=[]
for t in sorted(set(r['target'] for r in qa)):
 for arm in ['cot','output']:
  rr=[r for r in qa if r['target']==t and r['arm']==arm and r['status']=='ok']; vals=[mod.compliance(r['control_type'],(r['reasoning'] if arm=='cot' else r['content']),r['grade_kwargs']) for r in rr];out['mvp_qa'].append({'target':t,'arm':arm,'n':len(rr),'nonempty':sum(bool(r['reasoning'] if arm=='cot' else r['content']) for r in rr),'passes':sum(v==1 for v in vals)})
dr=rows(BASE/'scratch_cotcontrol/out/rows_depth_main.jsonl');out['kimi_depth']=[]
for h in sorted(set(r['h'] for r in dr)):
 rr=[r for r in dr if r['target']=='kimi-k3' and r['h']==h and r['control_type']=='lowercase_thinking'];bb=[r for r in dr if r['target']=='kimi-k3' and r['h']==h and r['arm']=='baseline']
 if rr:out['kimi_depth'].append({'h':h,'n':len(rr),'passes':sum(mod.compliance(r['control_type'],r['reasoning'],r['grade_kwargs'])==1 for r in rr),'baseline_n':len(bb),'baseline_correct':sum(bool(re.search(r'Answer:\s*'+re.escape(str(r['correct_answer']))+r'\b',r['content'] or '',re.I)) for r in bb)})
br=rows(BASE/'scratch_cotcontrol/out/rows_brew_suppress.jsonl');out['qwen_brew']=[]
for arm in sorted(set(r['control_type'] for r in br),key=str):
 rr=[r for r in br if r['target']=='qwen3.8-2.4t' and r['control_type']==arm]
 if rr:out['qwen_brew'].append({'type':arm,'n':len(rr),'passes':sum(mod.compliance(r['control_type'],r['reasoning'],r['grade_kwargs'])==1 for r in rr)})
(OUT/'checks.json').write_text(json.dumps(out,indent=2,default=lambda x:float(x)))
# Every report's question, section inventory, and all numeric tables, with original line references.
inv={}
for p in sorted((BASE/'results/report').glob('*.md')):
 lines=p.read_text(errors='replace').splitlines();inv[p.name]={'path':str(p),'lines':len(lines),'headings':[{'line':i+1,'text':s} for i,s in enumerate(lines) if s.startswith('#')],'tables':[{'line':i+1,'text':s} for i,s in enumerate(lines) if s.startswith('|')]}
(OUT/'report_inventory.json').write_text(json.dumps(inv,indent=2))
print(json.dumps(out,indent=2,default=lambda x:float(x)))
