exec(open('/home/john/code/results/neel-astra/work/sol-crosscheck/setup/check_setup.py').read().split('# Validate source')[0].replace("print(json.dumps({k:v for k,v in result.items() if k not in ['john_fields','neel_raw','john_x']},indent=2));print('x graded files',len(x));print('john fields',result['john_fields'])",''))
import sys
sys.path.insert(0,str(J/'grading'));from cotswitch import data as jd
q=lambda r:jd.load_items(r['dataset'])[r['item_idx']].question
by={d:len({q(r) for r in jrows if r['dataset']==d}) for d in ['gpqa','hle','mmlu_pro']}
print('question only by ds',by)
print('john model names',sorted({m for xx in x for m in xx['models']}))
print('neel GLM raw',rawstats['z-ai/glm-5.3'])
print('neel fallback counts',{k:dict(v['fallbacks']) for k,v in rawstats.items() if any(a!='None' for a in v['fallbacks'])})
print('John initial sampling distinct',len({json.dumps(r['sampling'],sort_keys=True) for r in jrows}),'system',collections.Counter(str(r['system_prompt']) for r in jrows))
# Check exact requirement match after replacing channel and using same control value, separately for every type.
from cotswitch import prompts as jp
import importlib.util
sp=importlib.util.spec_from_file_location('pp','/home/john/code/cot-control-ref/CoT-Control-QA/prompts.py'); pp=importlib.util.module_from_spec(sp);sp.loader.exec_module(pp)
print('all mode templates match',pp.CONTROL_TYPE_REQUIREMENTS==jp.CONTROL_TYPE_REQUIREMENTS)
