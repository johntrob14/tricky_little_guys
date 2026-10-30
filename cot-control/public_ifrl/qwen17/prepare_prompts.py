import importlib.util,json,sys,hashlib,ast
from pathlib import Path
source=Path('/home/john/code/tlg-inksm/cot-control/results/outctl/elicit/elicit.py')
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading']
spec=importlib.util.spec_from_file_location('elicitation_source',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=Path(__file__).parent
assert len(m.ITEMS)==108
rows=[]
for it in m.ITEMS:
 ga=it['grader_args'];ga=ast.literal_eval(ga) if isinstance(ga,str) else dict(ga)
 for v in ['base','shot4','shotstack']:
  system,hist,user=m.build(it,v,'qwen')
  rows.append(dict(key=it['key'],dataset=it['dataset'],item_idx=it['item_idx'],constraint=it['constraint'],gold=it['answer_gold'],grader_args=ga,variant=v,system=system,hist=hist,user=user))
assert len(rows)==324 and len({(r['key'],r['variant']) for r in rows})==324
(out/'prompts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(out/'prompt_provenance.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),astra_rows=108,prompt_conditions=['base','shot4','shotstack'],units=len(rows),prompt_file_sha256=hashlib.sha256((out/'prompts.jsonl').read_bytes()).hexdigest()),indent=2))
print(json.dumps(json.loads((out/'prompt_provenance.json').read_text())))
