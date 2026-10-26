import json,pathlib,collections,re,random,unicodedata
import pandas as pd
O=pathlib.Path(__file__).parent
N=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control')
J=pathlib.Path('/home/john/code/tlg-arena-rl/cot-control')
raw={}; grades={}
for p in (N/'rows').glob('*.jsonl'):
 for l in p.open():
  r=json.loads(l)
  if r['status']=='ok' and not r.get('salt'):raw[r['model'],r['item_id']]=r
for p in (N/'graded').glob('*.jsonl'):
 for l in p.open():
  r=json.loads(l)
  if not r.get('salt'):grades[r['model'],r['item_id']]=r
out={'raw_n':len(raw),'caps_all':sum(r['finish_reason']=='length' for r in raw.values()),'empty':sum(not r['reasoning'].strip() for r in raw.values()),'passes_below20_latin':sum(g['compliance']==1 and len(re.findall('[A-Za-z]',raw[k]['reasoning']))<20 for k,g in grades.items()),'cjk_traces':sum(any('\u4e00'<=c<='\u9fff' for c in r['reasoning']) for r in raw.values())}
# Check exact question identity using source CSV rather than matching stripped prompt templates.
csvs={d:pd.read_csv(pathlib.Path('/home/john/code/cot-control-ref/CoT-Control-QA/datasets')/f) for d,f in [('gpqa','gpqa_w_keyword.csv'),('hle','hle_w_keyword.csv'),('mmlu_pro','mmlu_pro_mini_w_keyword.csv')]}
nr=[r for r in raw.values() if r['model']=='moonshotai/kimi-k3']
jr=[json.loads(l) for l in (J/'results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl').open()]
def jquestion(r):
 p=r['user_prompt']; q=p[len('Question: '):]; return q.split('\n\nA)')[0]
nq={(r['dataset'],str(csvs[r['dataset']].iloc[int(r['qid'])]['question'])) for r in nr}
jq={(r['dataset'],jquestion(r)) for r in jr}
ni={(r['dataset'],str(csvs[r['dataset']].iloc[int(r['qid'])]['question']),r['mode']) for r in nr}
ji={(r['dataset'],jquestion(r),r['constraint']) for r in jr}
out['question_match']={'neel_unique':len(nq),'john_unique':len(jq),'intersection':len(nq&jq),'same_constraint_intersection':len(ni&ji),'datasets':collections.Counter(d for d,q in nq&jq)}
# Validate output overlap counts directly.
out['glm_overlap']={}
for arm in ['cot','cot_low','expl']:
 p=J/f'results/outctl/xfull/{arm}/glm-base-or/graded.json'
 rs=json.loads(p.read_text());out['glm_overlap'][arm]={'n':len(rs),'strict':sum(r['strict'] for r in rs)}
# Re-read 12 positive and 12 negative cases, deterministic; save short private excerpts.
review=[]
for label in [1,0]:
 keys=sorted(k for k,g in grades.items() if g['compliance']==label)
 for k in random.Random(20261026).sample(keys,12):
  r=raw[k];review.append({'model':k[0],'item_id':k[1],'compliance':label,'mode':r['mode'],'prefix':r['reasoning'][:250],'suffix':r['reasoning'][-250:],'control_value':r['control_value'],'chars':len(r['reasoning'])})
(O/'review.json').write_text(json.dumps(review,indent=2))
(O/'more.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
