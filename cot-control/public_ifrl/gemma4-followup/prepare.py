import importlib.util,json,sys,hashlib
from pathlib import Path
src=Path('/home/john/code/tlg-inksm/cot-control/results/outctl/elicit/elicit.py')
sys.path[:0]=['/home/john/code/nocotrl-b-pilot/cot-control/rl','/home/john/code/nocotrl-b-pilot/cot-control/grading',str(src.parent)]
spec=importlib.util.spec_from_file_location('existing_elicitation',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=Path(__file__).parent
selected={}
for it in sorted(m.ITEMS,key=lambda r:r['key']):selected.setdefault(it['constraint'],it)
rows=[]
for c,it in sorted(selected.items()):
 for target in ['cot','output']:
  for elicitation in ['base','shot4']:
   if target=='cot':
    sy,hist,user=m.build(it,elicitation,'qwen')
    # Gemma's channel name is thought, not Qwen's think tags. No output text prefill.
    old='thinking (everything inside your <think> ... </think> block, before your answer)'
    def fix(s):return s.replace(old,'thought channel (your private reasoning before your final answer)').replace('reasoning stage','thought channel')
    sy=fix(sy) if sy else None;user=fix(user);hist=[(fix(u),t,a) for u,t,a in hist]
   else:
    from xmodel import expl_prompt
    sy=None;user=expl_prompt(it);hist=[]
    if elicitation=='shot4':
     hist=[(s['q']+('' if c=='ignore_question' else '\n\nState your answer, and then explain it.')+'\n\nRequirement: '+m.shot_req_out(c,s),'',m.shot_resp_out(c,s)) for s in m.SHOTS]
   rows.append(dict(key=it['key'],dataset=it['dataset'],item_idx=it['item_idx'],constraint=c,gold=it['answer_gold'],grader_args=it['grader_args'],target=target,elicitation=elicitation,system=sy,hist=hist,user=user))
assert len(rows)==36
(out/'prompts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(out/'protocol.json').write_text(json.dumps(dict(stage='pilot only; review before scaling',model='google/gemma-4-31b-it',hf_model='google/gemma-4-31B-it',hf_revision='842da3794eaa0b77d5f08bae87a17459d91ff475',provider='Crusoe',provider_quantization='bf16',prompt_rate_per_token=.00000014,completion_rate_per_token=.0000004,rows=36,temperature=1.0,top_p=1.0,max_tokens=8000,seed=13,budget_usd=1.0,prior_runner_issue='HF rendering branch did not enable thinking; native Gemma4 template defaults to false and drops ordinary historical reasoning',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),prompt_sha256=hashlib.sha256((out/'prompts.jsonl').read_bytes()).hexdigest(),after='Qwen full sweep ALL_DONE and all eight checkpoint exports verified/pushed',question='How do output control and thought-channel control differ under baseline and four-shot elicitation?',prediction='A persistent elicited output-vs-thought gap supports an application bottleneck; narrowing under demonstrations favors elicitation limits. This pilot only validates modes and observability.'),indent=2))
print('prepared',len(rows),'pilot cells')
