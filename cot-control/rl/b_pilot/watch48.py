"""Evaluate the exact step48 checkpoint concurrently, separate $0.50 ledger."""
import asyncio,json,sys,time,statistics
from pathlib import Path
from types import SimpleNamespace
P=Path('/home/john/code/results/nocotrl-b-pilot');O=P/'checkpoint48';R=Path('/home/john/code/nocotrl-b-pilot');B=R/'cot-control/rl/b_pilot'
sys.path[:0]=[str(B),str(R/'cot-control/grading'),str(R/'cot-control/rl')]
from pilot import API,evaluate_dev,evaluate_astra,readj
from analyze import independent_output
from dotenv import load_dotenv
from cotswitch import graders as G
from astra_continuous import score

def status(stage,**kw):(O/'status.json').write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2))
async def main():
 O.mkdir(exist_ok=True);status('waiting_for_step48')
 for _ in range(1440):
  try:ck=json.load(open(P/'checkpoints.json'))
  except json.JSONDecodeError:await asyncio.sleep(5);continue
  found=[c for c in ck if c['step']==48]
  if found:break
  await asyncio.sleep(5)
 else:raise TimeoutError('No step48 checkpoint within two hours')
 (O/'config.json').write_text(json.dumps(dict(checkpoint=found[0],incremental_budget=.5,combined_ceiling_note='Main evaluator capped at18; independent48 capped at0.50, below original20 total.',evaluation='60dev and108Astra; original prompts, same sampling settings.'),indent=2))
 (O/'dev.jsonl').write_bytes((P/'dev.jsonl').read_bytes())
 load_dotenv(Path.home()/'.config/sauce/tinker.env');api=API(O);api.ledger.limit=.5;api.ledger.phase_limit=.5;api.ledger.phase='eval48';api.sem=asyncio.Semaphore(1)
 await api.initialize();client=await api.service.create_sampling_client_async(model_path=found[0]['sampler'])
 status('evaluating',checkpoint=found[0]['sampler'])
 await evaluate_dev(api,client,'s48');await evaluate_astra(api,client,'s48',SimpleNamespace(repo=str(R)))
 tasks={r['id']:r for r in readj(O/'dev.jsonl')};dev=[independent_output(tasks[r['task_id']],r) for r in readj(O/'dev_s48.jsonl') if tasks[r['task_id']]['rule']!='none'];a=[]
 for r in readj(O/'astra_full_s48.jsonl'):
  st=G.grade('astra',r['constraint'],r['cot'],r['grader_args'])['strict']
  if st is not None:a.append(dict(constraint=r['constraint'],strict=st,partial=score(dict(constraint=r['constraint'],reasoning=r['cot'],grader_args=r['grader_args']),st)))
 result=dict(dev_n=len(dev),dev_strict=sum(r['strict_robust'] for r in dev),dev_joint=sum(r['joint_robust'] for r in dev),astra_n=len(a),astra_strict=sum(r['strict'] for r in a),astra_partial=statistics.mean(r['partial'] for r in a),incremental_cost_upper=api.ledger.committed)
 (O/'summary.json').write_text(json.dumps(result,indent=2));status('done',**result)
 with open(P/'sprint_log_pending.txt','a') as f:f.write('\nStep48 concurrent evaluation complete: checkpoint48/summary.json. Separate ledger must be added to main costs; raw trace audit pending.\n')
if __name__=='__main__':
 try:asyncio.run(main())
 except BaseException as e:status('failed',error=repr(e));raise
