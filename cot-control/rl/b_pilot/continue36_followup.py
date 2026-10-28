"""Follow the running continuation, then evaluate and recompute its result."""
import json,os,subprocess,sys,time
from pathlib import Path
P=Path('/home/john/code/results/nocotrl-b-pilot');R=Path('/home/john/code/nocotrl-b-pilot');B=R/'cot-control/rl/b_pilot'
def status(stage,**kw):
 (P/'continuation36_status.json').write_text(json.dumps(dict(stage=stage,time=time.time(),pid=os.getpid(),**kw),indent=2))
def main():
 status('waiting_for_step36')
 for _ in range(720):
  ck=json.load(open(P/'checkpoints.json'))
  if ck[-1]['step']==36:break
  if ck[-1]['step']>36:raise RuntimeError('Unexpected checkpoint beyond 36; refusing to evaluate a different model.')
  time.sleep(5)
 else:raise TimeoutError('Checkpoint36 not saved within one hour; inspect training logs.')
 for stage,cmd in [('evaluation',[str(B/'pilot.py'),'eval','--label','s36','--out',str(P),'--repo',str(R)]),('analysis',[str(B/'continuation_summary.py')])]:
  status(stage)
  with open(P/f'continuation36_{stage}.log','a') as f:subprocess.run([sys.executable,*cmd],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=R)
 result=json.load(open(P/'continuation36_summary.json'));status('done',cost_upper=result['cost_upper'],summary=str(P/'continuation36_summary.json'))
 with open(P/'sprint_log_pending.txt','a') as f:f.write('\nContinuation to36 completed, raw-output comparisons in continuation36_summary.json; cost upper $'+str(result['cost_upper'])+'. Raw audit sample saved; human trace review still pending.\n')
if __name__=='__main__':
 try:main()
 except BaseException as e:status('failed',error=repr(e));raise
