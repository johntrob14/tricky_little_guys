import json, pathlib, collections, hashlib, csv, ast
O=pathlib.Path(__file__).parent
N=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control')
J=pathlib.Path('/home/john/code/tlg-arena-rl/cot-control')
nrows=[json.loads(l) for l in (N/'rows/moonshotai_kimi-k3.jsonl').open()]
nrows={r['item_id']:r for r in nrows if r['status']=='ok' and r.get('salt','')==''}
jrows=[json.loads(l) for l in (J/'results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl').open()]
def question(p):return p.split('\n\nRequirement:')[0].split('\n\n\nYour response')[0]
nq={(r['dataset'],question(r['prompt'])) for r in nrows.values()}
jq={(r['dataset'],question(r['user_prompt'])) for r in jrows}
ni={(r['dataset'],question(r['prompt']),r['mode']) for r in nrows.values()}
ji={(r['dataset'],question(r['user_prompt']),r['constraint']) for r in jrows}
match=[]
for r in nrows.values():
 matches=[x for x in jrows if x['dataset']==r['dataset'] and question(x['user_prompt'])==question(r['prompt'])]
 match.append(dict(neel_item=r['item_id'],question_sha256=hashlib.sha256(question(r['prompt']).encode()).hexdigest(),john_question_matches=[x['key'] for x in matches],john_same_type=[x['key'] for x in matches if x['constraint']==r['mode']]))
(O/'matches.json').write_text(json.dumps(match,indent=2))
req={m:dict(neel=next(r['prompt'].split('\n\nRequirement:')[1] for r in nrows.values() if r['mode']==m),john=next(r['user_prompt'].split('\n\nRequirement:')[1] for r in jrows if r['constraint']==m)) for m in sorted({r['mode'] for r in nrows.values()})}
(O/'requirements.json').write_text(json.dumps(req,indent=2))
rawstats={}
for p in (N/'rows').glob('*.jsonl'):
 rs={}
 for l in p.open():
  r=json.loads(l)
  if r['status']=='ok' and not r.get('salt'):rs[r['item_id']]=r
 rawstats[next(iter(rs.values()))['model']]=dict(n=len(rs),fallbacks=collections.Counter(str(r.get('fallback')) for r in rs.values()),providers=collections.Counter(r['provider'] for r in rs.values()),caps=sum(r['finish_reason']=='length' for r in rs.values()))
x=[]
for p in (J/'results/outctl').glob('x*/*/*/graded.json'):
 rs=json.load(p.open());x.append(dict(path=str(p),n=len(rs),models=sorted(set(str(r.get('model')) for r in rs)),strict=sum(r['strict'] for r in rs),types=collections.Counter(r['constraint'] for r in rs)))
for p in (J/'results/outctl').glob('glm_snapshot*/cot/graded.json'):
 rs=json.load(p.open());x.append(dict(path=str(p),n=len(rs),models=sorted(set(str(r.get('model')) for r in rs)),strict=sum(r['strict'] for r in rs),types=collections.Counter(r['constraint'] for r in rs)))
result=dict(neel_items=len(ni),john_items=len(ji),neel_questions=len(nq),john_questions=len(jq),shared_questions=len(nq&jq),shared_same_type_items=len(ni&ji),neel_rows_with_shared_question=sum((r['dataset'],question(r['prompt'])) in jq for r in nrows.values()),by_dataset={d:dict(neel_questions=sum(a==d for a,b in nq),john_questions=sum(a==d for a,b in jq),shared_questions=sum(a==d for a,b in nq&jq),shared_items=sum(a==d for a,b,c in ni&ji)) for d in ['gpqa','hle','mmlu_pro']},john_fields={k:v for k,v in jrows[0].items() if k not in ['user_prompt','rendered_prompt','reasoning','response','raw_output','grader_args','question']},neel_raw=rawstats,john_x=x)
(O/'checks.json').write_text(json.dumps(result,indent=2,default=str))
print(json.dumps({k:v for k,v in result.items() if k not in ['john_fields','neel_raw','john_x']},indent=2));print('x graded files',len(x));print('john fields',result['john_fields'])
# Validate source templates and every stored requirement, independent of item sample.
import importlib.util,sys
spec=importlib.util.spec_from_file_location('paper_prompts','/home/john/code/cot-control-ref/CoT-Control-QA/prompts.py'); pp=importlib.util.module_from_spec(spec);spec.loader.exec_module(pp)
sys.path.insert(0,str(J/'grading'))
from cotswitch import prompts as jp,data as jd
checks=dict(template_dict_equal=pp.CONTROL_TYPE_REQUIREMENTS==jp.CONTROL_TYPE_REQUIREMENTS,neel_requirements_matching_paper=0,john_requirements_matching_builder=0,neel_requirement_n=len(nrows),john_requirement_n=len(jrows))
for r in nrows.values():
 exp=pp._get_requirement_text(r['mode'],r['control_value'],'reasoning stage',pp.CONTROL_TYPE_REQUIREMENTS,r['model'])
 checks['neel_requirements_matching_paper']+=r['prompt'].split('\n\nRequirement: ')[1]==exp
for r in jrows:
 it=jd.load_items(r['dataset'])[r['item_idx']]
 exp=jp.requirement_text(r['constraint'],jp.get_control_value(r['constraint'],it)).replace('reasoning stage','analysis channel')
 checks['john_requirements_matching_builder']+=r['user_prompt'].split('\n\nRequirement: ')[1]==exp
# Question-only exact comparison via released CSVs, independent of full prompt matching.
qN={(r['dataset'],jd.load_items(r['dataset'])[r['qid']].question) for r in nrows.values()}
qJ={(r['dataset'],jd.load_items(r['dataset'])[r['item_idx']].question) for r in jrows}
checks.update(question_only_neel=len(qN),question_only_john=len(qJ),question_only_overlap=len(qN&qJ),overlap_indices=[dict(neel=r['item_id'],john=[x['key'] for x in jrows if x['dataset']==r['dataset'] and x['item_idx']==r['qid']]) for r in nrows.values() if any(x['dataset']==r['dataset'] and x['item_idx']==r['qid'] for x in jrows)])
# Raw exact model comparison, keeping arms and programmatic slice separate.
glm=[]
for side in ['cot','cot_low','expl']:
 p=J/f'results/outctl/xfull/{side}/glm-base-or/graded.json';rs=json.load(p.open());pr=[r for r in rs if r['constraint']!='ignore_question']
 glm.append(dict(side=side,n=len(rs),passes=sum(r['strict'] for r in rs),programmatic_n=len(pr),programmatic_passes=sum(r['strict'] for r in pr),providers=collections.Counter(str(r.get('meta',{}).get('provider')) for r in rs),caps=sum(r.get('stop')=='length' for r in rs),models=collections.Counter(r['model'] for r in rs)))
ng={r['item_id']:r for r in map(json.loads,(N/'graded/z-ai_glm-5.3.jsonl').open()) if not r.get('salt')}
checks['glm_neel']=dict(n=len(ng),passes=sum(r['compliance']==1 for r in ng.values()),programmatic_n=sum(r['mode']!='ignore_question' for r in ng.values()),programmatic_passes=sum(r['compliance']==1 and r['mode']!='ignore_question' for r in ng.values()))
checks['glm_john']=glm
(O/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
