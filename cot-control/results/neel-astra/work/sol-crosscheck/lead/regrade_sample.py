import pathlib,json,sys,collections,random
O=pathlib.Path(__file__).parent
N=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control')
sys.path.insert(0,'/home/john/code/cot-control-ref/CoT-Control-QA')
sys.path.insert(0,'/home/john/code/tlg-arena-rl/cot-control/grading')
import grading as ref
from cotswitch.graders import _grade_default
from cotswitch.data import load_items
from cotswitch.prompts import _default_grader_args
import re
items={(d,it.item_idx):it for d in ['gpqa','hle','mmlu_pro'] for it in load_items(d)}
groups=collections.defaultdict(list)
for p in (O.parent/'regrade/row_results.csv').open(): pass
import csv
for r in csv.DictReader((O.parent/'regrade/row_results.csv').open()):groups[r['model'],r['mode'],r['neel']].append(r)
out=[]; cache={}
for key,rs in sorted(groups.items()):
 rr=random.Random(20261026).choice(rs)
 if rr['raw_file'] not in cache:
  cache[rr['raw_file']]={r['item_id']:r for r in map(json.loads,pathlib.Path(rr['raw_file']).read_text().splitlines()) if r['status']=='ok' and not r.get('salt')}
 r=cache[rr['raw_file']][rr['item_id']]
 a=ref.grade_cot_compliance(r['prompt'],r['reasoning']);b=_grade_default(r['mode'],r['reasoning'],{'user_prompt':r['prompt']})['strict']
 args=_default_grader_args(items[r['dataset'],r['qid']],r['mode'],r['prompt'])
 args={k:v for k,v in args.items() if k not in ['prompt','user_prompt']}
 body=re.sub(r'(<\|(return|end|call)\|>)+$','',r['reasoning']).strip()
 c=_grade_default(r['mode'],body,args)['strict'] if body else 0
 out.append({'model':r['model'],'item_id':r['item_id'],'mode':r['mode'],'neel':int(rr['neel']),'ref':a,'john':b,'john_kwargs':c})
summary={'n':len(out),'models':len({r['model'] for r in out}),'modes':len({r['mode'] for r in out}),'ref_agree':sum(r['neel']==r['ref'] for r in out),'john_agree':sum(r['neel']==r['john'] for r in out),'positives':sum(r['neel'] for r in out)}
summary['john_kwargs_agree']=sum(r['neel']==r['john_kwargs'] for r in out)
(O/'sample_results.json').write_text(json.dumps({'summary':summary,'rows':out},indent=2)); print(summary)
