import json,os,subprocess,sys,time,shutil
from pathlib import Path
P=Path('/home/john/code/results/nocotrl-b-pilot');O=P/'additive';R=Path('/home/john/code/nocotrl-b-pilot');B=R/'cot-control/rl/b_pilot'
def status(stage,**kw):(O/'status.json').write_text(json.dumps(dict(stage=stage,pid=os.getpid(),time=time.time(),**kw),indent=2))
def cost(p):
 active={};spent=0
 if not p.exists():return 0
 for line in open(p):
  r=json.loads(line)
  if r['event']=='reserve':active[r['id']]=r['usd']
  else:active.pop(r['id']);spent+=r['usd']
 return spent+sum(active.values())
def main():
 O.mkdir(exist_ok=True);assert not (O/'costs.jsonl').exists();status('waiting_for_previous_eval')
 for _ in range(720):
  old=json.load(open(P/'continuation60_status.json'));extra=json.load(open(P/'checkpoint48/status.json'))
  if old['stage'] in ['done','failed'] and extra['stage'] in ['done','failed']:break
  time.sleep(5)
 else:raise TimeoutError('Previous evaluators still active')
 previous=cost(P/'costs.jsonl')+cost(P/'checkpoint48/costs.jsonl');budget=19-previous;assert budget>2
 cfg=dict(initialization='Fresh base, LoRA rank32, seed271027',learning_rate=4e-5,reward='1.5*task_success + partial + 0.5*strict',budget_usd=budget,previous_cost_upper=previous,total_cap=20,storage_reserve=1,steps=60,evaluate_every=12,data='Identical480 training tasks; repeated with fresh rollouts; all Astra questions held out',changes=['Additive reward','Save available forward/backward and optimizer metrics'],prediction='Restoring formatting feedback on task failures increases informative groups and CoT transfer.',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip())
 (O/'additive_config.json').write_text(json.dumps(cfg,indent=2))
 for n in ['train.jsonl','dev.jsonl','data_manifest.json']:shutil.copyfile(P/n,O/n)
 with open(P/'sprint_log_pending.txt','a') as f:f.write('\nFresh additive B restart: additive/additive_config.json. Same data and LR; reward1.5task+partial+0.5strict; optimizer metrics retained; eval every12steps.\n')
 for end in [12,24,36,48,60]:
  train=['additive.py','train','--end-step',str(end),'--out',str(O),'--repo',str(R)]
  if end>12:train+=['--resume','--segment',f'to{end}','--resume-reason','Scheduled additive continuation after checkpoint evaluation.']
  ev=['additive.py','eval','--label',f's{end}','--out',str(O),'--repo',str(R)]
  for stage,args in [('training',train),('evaluation',ev)]:
   status(stage,step_target=end)
   with open(O/f'{stage}_{end}.log','a') as f:subprocess.run([sys.executable,str(B/args[0]),*args[1:]],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=R,check=True)
   if stage=='training':assert json.load(open(O/'checkpoints.json'))[-1]['step']==end,'Stopped before target; inspect budget'
  status('checkpoint_evaluated',step=end,cost_upper_combined=previous+cost(O/'costs.jsonl'))
 status('done',cost_upper_combined=previous+cost(O/'costs.jsonl'))
if __name__=='__main__':
 O.mkdir(exist_ok=True)
 try:main()
 except BaseException as e:status('failed',error=repr(e));raise
