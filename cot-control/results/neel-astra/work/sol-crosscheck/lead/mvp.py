import json,pathlib,sys,collections
sys.path.insert(0,'/home/john/code/cot-control-ref/CoT-Control-QA')
import grading as g
P=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cotcontrol/out')
O=pathlib.Path(__file__).parent
def read(name):return [json.loads(l) for l in (P/name).open()]
def grade(r,text):
 k=r['grade_kwargs'];t=r['control_type']
 if t=='repeat_sentences':return g.grade_repeat_sentences(text,k['target_sentence'])
 if t in ['word_suppression','word_suppression_placebo']:return g.grade_word_suppression(text,k['keyword'],k.get('synonyms') or [])
 return g.GRADE_FUNC_MAP[t](text)
qa=collections.defaultdict(lambda:[0,0])
for r in read('rows_qa_mvp.jsonl'):
 if r['status']!='ok' or r['arm'] not in ['cot','output']:continue
 text=r['reasoning'] if r['arm']=='cot' else r['content'];v=qa[r['target']+' '+r['arm']];v[0]+=grade(r,text);v[1]+=1
depth=collections.defaultdict(lambda:[0,0])
for r in read('rows_depth_main.jsonl'):
 if r['target']=='kimi-k3' and r['control_type']=='lowercase_thinking':
  v=depth[r['h']];v[0]+=g.grade_lowercase_thinking(r['reasoning']);v[1]+=1
brew=collections.defaultdict(lambda:[0,0])
for r in read('rows_brew_suppress.jsonl'):
 if r['target']=='qwen3.8-2.4t' and r['control_type'] in ['word_suppression','word_suppression_placebo']:
  v=brew[r['control_type']];v[0]+=grade(r,r['reasoning']);v[1]+=1
out={'qa':qa,'kimi_depth':depth,'qwen_brew':brew};(O/'mvp.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
