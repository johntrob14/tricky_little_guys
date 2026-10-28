"""Run authorized steps37–60 and identical evaluation under the shared budget ledger."""
import json,os,subprocess,sys,time
from pathlib import Path
P=Path('/home/john/code/results/nocotrl-b-pilot');R=Path('/home/john/code/nocotrl-b-pilot');B=R/'cot-control/rl/b_pilot'
def status(stage,**kw):
 (P/'continuation60_status.json').write_text(json.dumps(dict(stage=stage,time=time.time(),pid=os.getpid(),**kw),indent=2))
def main():
 ck=json.load(open(P/'checkpoints.json'));assert ck[-1]['step']==36
 assert not (P/'steps_continuation60.jsonl').exists()
 cfg=dict(start=36,end=60,source_checkpoint=ck[-1],budget_total=20,expected_incremental_usd=2,training_absolute_ceiling=10,evaluation_absolute_ceiling=18,recipe='Same fixed480-task pool, fresh on-policy rollouts, same reward/lr/group size; concurrency3; no Astra in training.',evaluation='Same60 dev and108 Astra prompts; original wording; compare base,24,36,60.',hypothesis='Additional training may reveal delayed CoT transfer as in original run; estimate partial and strict compliance separately.',time=time.time())
 (P/'continuation60_config.json').write_text(json.dumps(cfg,indent=2))
 with open(P/'sprint_log_pending.txt','a') as f:f.write('\nRunning continuation36→60 with same recipe and data, shared $20 cap; hypothesis: delayed CoT transfer. See continuation60_config.json.\n')
 for stage,args in [('training',['pilot.py','train','--resume','--segment','continuation60','--end-step','60','--out',str(P),'--repo',str(R)]),('evaluation',['pilot.py','eval','--label','s60','--out',str(P),'--repo',str(R)]),('analysis',['continuation60_summary.py'])]:
  status(stage)
  with open(P/f'continuation60_{stage}.log','a') as f:subprocess.run([sys.executable,str(B/args[0]),*args[1:]],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=R)
  if stage=='training':assert json.load(open(P/'checkpoints.json'))[-1]['step']==60
 result=json.load(open(P/'continuation60_summary.json'));status('done',cost_upper=result['cost_upper'])
 with open(P/'sprint_log_pending.txt','a') as f:f.write('\nContinuation60 complete; raw comparisons in continuation60_summary.json. Trace review pending.\n')
if __name__=='__main__':
 try:main()
 except BaseException as e:status('failed',error=repr(e));raise
